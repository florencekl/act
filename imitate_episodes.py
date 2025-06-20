import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from deepdrr import LineAnnotation
from deepdrr import geo
import killeengeo as kg
import torch
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

import h5py

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from constants import ACTION_DIM
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

from deepdrr_simulation_platform import load_config, SimulationEnvironment
from deepdrr.utils import image_utils

import IPython
e = IPython.embed

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class'] if 'policy_class' in args else 'ACT'
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size'] if 'batch_size' not in args else 100 # batch size should be smaller than the number of recorded actions
    batch_size_val = args['batch_size'] if 'batch_size' not in args else 100 # batch size should be smaller than the number of recorded actions
    num_epochs = args['num_epochs'] if 'num_epochs' in args else 1000
    action_dim = args['action_dim'] if 'action_dim' in args else 9

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    # we always wanna use our task configs in this repo, no outside dependencies
    from constants import SIM_TASK_CONFIGS
    task_config = SIM_TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    # TODO change state_dim according to task
    state_dim = action_dim
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'action_dim': action_dim
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names, 'action_dim': action_dim}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    print(is_eval)
    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        # ckpt_names = [f'policy_epoch_300_seed_0.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image



def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    ### --- SETUP SIMULATION ENVIRONMENT --- ###
    file = h5py.File("/data2/flora/vertebroplasty_imitation_0/episode_190.hdf5", 'r')
    
    # Load metadata
    case = file.attrs['case']
    phantoms_dir = file.attrs['phantoms_dir'] 
    qpos_h5data = file['observations/qpos'][:]
    
    # Load annotation data
    start_point = file['annotations/start'][:]
    end_point = file['annotations/end'][:]
    world_from_anatomical = file['world_from_anatomical'][:]
    
    print(f"Case: {case}")
    print(f"Frames: {len(qpos_h5data)}")
    print(f"QPos shape: {qpos_h5data.shape}")
    
    # Setup environment
    cfg = load_config("/home/flora/projects/verteboplasty_imitation/deepdrr_simulation_platform/config.yaml")
    env = SimulationEnvironment(cfg)
    
    # Load phantom and tools
    phantom_path = os.path.join(phantoms_dir, case)
    print(f"Loading phantom from: {phantom_path}")
    ct = env.load_phantom(phantom_path)
    print(f"Loading tools for case: {case}")
    tools = env.load_tools()
    print(f"Initializing projector")
    env.initialize_projector()
    
    # Create annotation object
    annotation = LineAnnotation(
        startpoint=geo.point(start_point[:3]),
        endpoint=geo.point(end_point[:3]),
        volume=ct,
        world_from_anatomical=ct.world_from_anatomical,
        # world_from_anatomical=kg.FrameTransform.identity(),
        anatomical_coordinate_system=ct.anatomical_coordinate_system
    )
    
    # Generate camera views
    ap_view, lateral_view = env.generate_views(annotation)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 5
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        lateral_images = []
        ap_images = []
        qpos_history = []

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        # qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()

        starting_timestep = rollout_id * 20
        # max_timesteps = max_timesteps - starting_timestep
        with torch.inference_mode():
            # get observation at start_ts only
            image_dict = dict()
            for cam_name in config['camera_names']:
                image_dict[cam_name] = file[f'/observations/images/{cam_name}'][starting_timestep]
                print(np.shape(image_dict[cam_name]), np.min(image_dict[cam_name]), np.max(image_dict[cam_name]))
            
            qpos = torch.from_numpy(qpos_h5data[starting_timestep]).float().numpy()

            for t in range(max_timesteps):
                # new axis for different cameras
                all_cam_images = []
                for cam_name in config['camera_names']:
                    all_cam_images.append(image_dict[cam_name])
                all_cam_images = np.stack(all_cam_images, axis=0)

                # construct observations
                image_data = torch.from_numpy(all_cam_images)

                # channel last
                image_data = torch.einsum('k h w c -> k c h w', image_data)

                # normalize image and change dtype to float
                image_data = image_data / 255.0

                image_data = image_data.float().cuda().unsqueeze(0)  # add batch dimension
                
                qpos = pre_process(qpos)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                # qpos_history[:, t] = qpos

                # print(image_data.shape, qpos.shape, action_data.shape, is_pad.shape)
                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, image_data)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, image_data)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ap_image = env.render_tools(
                    tool_poses={
                        "cannula": (geo.point(target_qpos[:3]), geo.vector(target_qpos[6:9])),
                        "linear_drive": (geo.point(target_qpos[3:6]), geo.vector(target_qpos[6:9])),
                    },
                    view=ap_view
                )
                # print(np.shape(ap_image), np.min(ap_image), np.max(ap_image))
                ap_processed = image_utils.process_drr(ap_image, neglog=False, invert=False, clahe=False)
                # print(np.shape(ap_processed), np.min(ap_processed), np.max(ap_processed))
                ap_images.append(ap_processed)
                
                # Generate lateral image  
                lateral_image = env.render_tools(
                    tool_poses={
                        "cannula": (geo.point(target_qpos[:3]), geo.vector(target_qpos[6:9])),
                        "linear_drive": (geo.point(target_qpos[3:6]), geo.vector(target_qpos[6:9])),
                    },
                    view=lateral_view
                )
                lateral_processed = image_utils.process_drr(lateral_image, neglog=False, invert=False)
                # print(np.shape(lateral_processed), np.min(lateral_processed), np.max(lateral_processed))
                lateral_images.append(lateral_processed)

                image_dict['ap'] = ap_processed
                image_dict['lateral'] = lateral_processed

                qpos = target_qpos

                qpos_history.append(qpos)

        # Convert to numpy arrays
        ap_images = np.array(ap_images)
        print(np.shape(ap_images), np.min(ap_images), np.max(ap_images))
        lateral_images = np.array(lateral_images)
        print(np.shape(lateral_images), np.min(lateral_images), np.max(lateral_images))
        
        print(f"Generated {len(ap_images)} images")
        print(f"AP image shape: {ap_images.shape}")
        print(f"Lateral image shape: {lateral_images.shape}")
        
        output_file = os.path.join(config['ckpt_dir'], f'{task_name}_eval_start_{starting_timestep}.hdf5')
        # Save new HDF5 file
        print(f"Saving to: {output_file}")
        
        with h5py.File(output_file, 'w') as new_f:
            # Copy all original attributes
            for key in file.attrs.keys():
                new_f.attrs[key] = file.attrs[key]
            
            # Mark as regenerated
            new_f.attrs['regenerated'] = True
            
            qpos = np.array(qpos_history)
            qvel = np.vstack(([0, 0, 0, 0, 0, 0, 0, 0, 0], np.diff(qpos, axis=0)))
            qvel = qvel.tolist()

            # Copy datasets
            new_f.create_dataset("world_from_anatomical", data=file['world_from_anatomical'][:])
            
            # Copy observations
            new_f.create_dataset("action", data=np.array(qpos, dtype=np.float32))
            obs_grp = new_f.create_group("observations")
            obs_grp.create_dataset("qpos", data=np.array(qpos, dtype=np.float32))
            obs_grp.create_dataset("qvel", data=np.array(qvel, dtype=np.float32))
            
            # Copy annotations
            anno_grp = new_f.create_group("annotations")
            anno_grp.create_dataset("start", data=np.array(annotation.startpoint_in_world.tolist()))
            anno_grp.create_dataset("end", data=np.array(annotation.endpoint_in_world.tolist()))
            
            # Add projection matrices
            proj_grp = obs_grp.create_group("projection_matrices")
            proj_grp.create_dataset("ap", data=np.array(file['observations/projection_matrices/ap']))
            proj_grp.create_dataset("lateral", data=np.array(file['observations/projection_matrices/lateral']))
            
            # Add regenerated images
            imgs_grp = obs_grp.create_group("images")
            imgs_grp.create_dataset("ap", data=ap_images)
            imgs_grp.create_dataset("lateral", data=lateral_images)

    return 0, 0


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    # print(f'Forward pass with {qpos_data.shape}, {image_data.shape}, {action_data.shape}, {is_pad.shape}')
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        # print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                # print(f'Validation batch {batch_idx}')
                # print(f'Batch data shapes: {data}')
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)
 
            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        # print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        # print(f"Summary_String: {summary_string}")

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        # print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        # print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    # parser.add_argument('--action_dim', action='store', type=int, help='Action Dimension', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))
