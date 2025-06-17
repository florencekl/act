import torch
import numpy as np
import os
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
    action_dim = args['action_dim'] if 'action_dim' in args else 6

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

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 1
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        file = h5py.File("/data2/flora/sim_vertebroplasty_simple/episode_1.hdf5", 'r')

        print(max_timesteps)

        current_timestep = 0
        with torch.inference_mode():
            for t in range(max_timesteps):
                

                original_action_shape = file['/action'].shape
                episode_len = original_action_shape[0]

                # get observation at start_ts only
                qpos = file['/observations/qpos'][current_timestep]
                qvel = file['/observations/qvel'][current_timestep]
                image_dict = dict()
                for cam_name in config['camera_names']:
                    image_dict[cam_name] = file[f'/observations/images/{cam_name}'][current_timestep]
                action = file['/action'][0:]
                action_len = episode_len - 0
                # else:
                #     action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                #     action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

                padded_action = np.zeros(original_action_shape, dtype=np.float32)
                padded_action[:action_len] = action
                is_pad = np.zeros(episode_len)
                is_pad[action_len:] = 1

                # new axis for different cameras
                all_cam_images = []
                for cam_name in config['camera_names']:
                    all_cam_images.append(image_dict[cam_name])
                all_cam_images = np.stack(all_cam_images, axis=0)

                # construct observations
                image_data = torch.from_numpy(all_cam_images)
                qpos_data = torch.from_numpy(qpos).float().numpy()
                action_data = torch.from_numpy(padded_action).float()
                is_pad = torch.from_numpy(is_pad).bool()

                # channel last
                image_data = torch.einsum('k h w c -> k c h w', image_data)

                # normalize image and change dtype to float
                image_data = image_data / 255.0

                image_data = image_data.float().cuda().unsqueeze(0)  # add batch dimension

                qpos = pre_process(qpos_data)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos

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

                print(f"t: {t}, target_qpos: {target_qpos}, qpos: {qpos_data}")

                target_qpos_list.append(target_qpos)

        print(len(target_qpos_list))
        import matplotlib.pyplot as plt

        from PIL import Image

        # --- Visualization: Projected Predicted Qpos for All 3 Views ---
        ap_imgs = np.array(file['observations/images/ap'])  # (N, H, W)
        lateral_imgs = np.array(file['observations/images/lateral'])
        # barrel_imgs = np.array(file['observations/images/barrel'])
        ap_proj = np.array(file['observations/projection_matrices/ap'])  # (3, 4)
        lat_proj = np.array(file['observations/projection_matrices/lateral'])
        # bar_proj = np.array(file['observations/barrel_projection_matrix'])

        def project_point(P, point3d):
            point_h = np.append(point3d, 1)
            proj = P @ point_h
            x = proj[0] / proj[2]
            y = proj[1] / proj[2]
            return x, y

        # --- Visualization: Overlay Predicted and Actual Qpos for All 3 Views ---
        actual_qpos = np.array(file['observations/qpos'])[current_timestep:]  # (N, 3)
        imgs = [ap_imgs[current_timestep], lateral_imgs[current_timestep]]
        projs = [ap_proj, lat_proj]
        view_names = ["ap", "lateral"]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        for ax, img, P, name in zip(axs, imgs, projs, view_names):
            if img.dtype != np.uint8:
                img = (255 * (img - img.min()) / (img.ptp() + 1e-8)).astype(np.uint8)
            ax.imshow(img, cmap='gray')
            # Plot predicted qpos
            pred_xs, pred_ys = [], []
            print(np.shape(target_qpos_list))
            for pt in target_qpos_list:
                x, y = project_point(P, pt[:3])
                pred_xs.append(x)
                pred_ys.append(y)
            ax.plot(pred_xs, pred_ys, 'rx', markersize=8, markeredgewidth=2, label='Predicted qpos', alpha=0.7)
            # Plot actual qpos
            act_xs, act_ys = [], []
            print(np.shape(actual_qpos))
            for pt in actual_qpos:
                x, y = project_point(P, pt[:3])
                act_xs.append(x)
                act_ys.append(y)
            ax.plot(act_xs, act_ys, 'bx', markersize=8, markeredgewidth=2, label='Actual qpos', alpha=0.7)
            ax.set_title(f"{name.upper()} View")
            ax.axis('off')
            ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, f'projected_pred_vs_actual_qpos_{rollout_id}.png'))
        plt.close()
        print(f"Saved projected_pred_vs_actual_qpos for rollout {rollout_id}")

    return 0, 0

    #         plt.close()
    #     if real_robot:
    #         move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
    #         pass

    #     rewards = np.array(rewards)
    #     episode_return = np.sum(rewards[rewards!=None])
    #     episode_returns.append(episode_return)
    #     episode_highest_reward = np.max(rewards)
    #     highest_rewards.append(episode_highest_reward)
    #     print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

    #     if save_episode:
    #         save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    # success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    # avg_return = np.mean(episode_returns)
    # summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    # for r in range(env_max_reward+1):
    #     more_or_equal_r = (np.array(highest_rewards) >= r).sum()
    #     more_or_equal_r_rate = more_or_equal_r / num_rollouts
    #     summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    # print(summary_str)

    # # save success rate to txt
    # result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    # with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
    #     f.write(summary_str)
    #     f.write(repr(episode_returns))
    #     f.write('\n\n')
    #     f.write(repr(highest_rewards))

    # return success_rate, avg_return


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
