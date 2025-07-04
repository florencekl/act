import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
import wandb
import platform

import h5py

from deepdrr_simulation_platform._generate_comparison_gif import triangulate_point
from utils import load_data # data functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy

from deepdrr_simulation_platform import generate_heatmap, load_config, SimulationEnvironment
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
    batch_size_train = args['batch_size'] if 'batch_size' in args else 8 # batch size should be smaller than the number of recorded actions
    batch_size_val = args['batch_size'] if 'batch_size' in args else 8 # batch size should be smaller than the number of recorded actions
    num_epochs = args['num_epochs'] if 'num_epochs' in args else 1000
    action_dim = args['action_dim'] if 'action_dim' in args and args['action_dim'] is not None else 11
    
    # wandb parameters
    use_wandb = args.get('use_wandb', False)
    wandb_project = args.get('wandb_project', 'vertebroplasty-imitation')
    wandb_entity = args.get('wandb_entity', None)

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
                         'action_dim': action_dim,
                         'input_channels': 3  # RGB + heatmap
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names, 'action_dim': action_dim, 'input_channels': 3}
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
        'real_robot': not is_sim,
        'use_wandb': use_wandb,
        'wandb_project': wandb_project,
        'wandb_entity': wandb_entity,
        'batch_size_train': batch_size_train,
        'batch_size_val': batch_size_val,
        'num_episodes': num_episodes,
        'resume_from_checkpoint': args.get('resume_from_checkpoint', None)
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        # ckpt_names = [f'policy_epoch_4300_seed_0.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            filenames = [
                "/data2/flora/vertebroplasty_imitation_2/episode_5001.hdf5",
                "/data2/flora/vertebroplasty_imitation_2/episode_5002.hdf5",
                "/data2/flora/vertebroplasty_imitation_2/episode_5003.hdf5",
                "/data2/flora/vertebroplasty_imitation_2/episode_5004.hdf5",
                "/data2/flora/vertebroplasty_imitation_2/episode_5005.hdf5",
                "/data2/flora/vertebroplasty_imitation_2/episode_5100.hdf5",
                "/data2/flora/vertebroplasty_imitation_2/episode_5175.hdf5",
            ]
            for name in filenames:
                success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, filename=name)
                results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()


    # --- Create a compact notes.md file in the checkpoint directory ---
    import sys, datetime, pprint
    os.makedirs(ckpt_dir, exist_ok=True)
    sysinfo = dict(
        python=platform.python_version(),
        pytorch=torch.__version__,
        cuda=torch.version.cuda if torch.cuda.is_available() else None,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        platform=platform.platform()
    )
    notes = f"""# Experiment Notes\n\n"""
    notes += f"Date: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
    notes += f"Command:\n```bash\npython3 {' '.join(sys.argv)}\n```\n"
    notes += f"Task: {task_name}\nCkpt dir: {ckpt_dir}\nPolicy: {policy_class}\nSeed: {args['seed']}\nBatch size: {batch_size_train}/{batch_size_val}\nEpochs: {num_epochs}\nLR: {args['lr']}\nAction dim: {action_dim}\nCameras: {camera_names}\n"
    notes += f"\nTask config:\n```\n{pprint.pformat(task_config)}\n```\n"
    notes += f"Args:\n```\n{pprint.pformat(args)}\n```\n"
    notes += f"System:\n```\n{pprint.pformat(sysinfo)}\n```\n"
    with open(os.path.join(ckpt_dir, 'notes.md'), 'w') as f:
        f.write(notes)

    # Initialize wandb for training
    if use_wandb:
        # Create run name with key parameters
        run_name = f"{policy_class}_{task_name}_lr{args['lr']}_bs{batch_size_train}_seed{args['seed']}"
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config=config,
            tags=[policy_class, task_name],
            save_code=True
        )
        # Log the config as a table for easy viewing
        wandb.config.update(config)
        # Create a table to track model architecture
        if policy_class == 'ACT':
            model_info = {
                'policy_class': policy_class,
                'hidden_dim': policy_config['hidden_dim'],
                'num_queries': policy_config['num_queries'],
                'enc_layers': policy_config['enc_layers'],
                'dec_layers': policy_config['dec_layers'],
                'nheads': policy_config['nheads'],
                'backbone': policy_config['backbone'],
                'action_dim': policy_config['action_dim'],
                'input_channels': policy_config['input_channels']
            }
        else:
            model_info = {
                'policy_class': policy_class,
                'backbone': policy_config['backbone'],
                'action_dim': policy_config['action_dim'],
                'input_channels': policy_config['input_channels']
            }
        wandb.log({"model_architecture": wandb.Table(
            columns=list(model_info.keys()),
            data=[list(model_info.values())]
        )}, commit=False)
        # Log system information
        system_info = {
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'platform': platform.platform()
        }
        if torch.cuda.is_available():
            system_info['gpu_name'] = torch.cuda.get_device_name(0)
        wandb.log({"system_info": wandb.Table(
            columns=list(system_info.keys()),
            data=[list(system_info.values())]
        )}, commit=False)
        # Log task configuration
        wandb.log({"task_config": wandb.Table(
            columns=list(task_config.keys()),
            data=[list(task_config.values())]
        )}, commit=False)
        # Log command line arguments
        wandb.log({"args": wandb.Table(
            columns=list(args.keys()),
            data=[list(args.values())]
        )}, commit=False)

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
    
    # Log final results to wandb
    if use_wandb:
        wandb.log({
            'final/best_epoch': best_epoch,
            'final/best_val_loss': min_val_loss
        })
        # Save model as wandb artifact
        artifact = wandb.Artifact(f"model_best_{run_name}", type="model")
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact)
        
        wandb.finish()


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



def eval_bc(config, ckpt_name, save_episode=True, filename=None):
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
    print(ckpt_path)
    print(policy_config)
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
    if filename is None:
        filename = "/data2/flora/vertebroplasty_imitation_1/episode_5001.hdf5"
    file = h5py.File(filename, 'r')
    
    # Load metadata
    case = file.attrs['case']
    phantoms_dir = file.attrs['phantoms_dir'] 
    qpos_h5data = file['observations/qpos'][:]
    
    # Load annotation data
    annotation_start = file['annotations/start'][:]
    annotation_end = file['annotations/end'][:]
    projection_matrices = {}
    for cam_name in config['camera_names']:
        projection_matrices[cam_name] = file[f'/observations/projection_matrices/{cam_name}'][:]
    world_from_anatomical = file['annotations/world_from_anatomical'][:]

    lateral_translation = file['device/lateral_translate'][()].item()
    superior_translation = file['device/superior_translate'][()].item()
    
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
        startpoint=geo.point(annotation_start[:3]),
        endpoint=geo.point(annotation_end[:3]),
        volume=ct,
        world_from_anatomical=ct.world_from_anatomical,
        # world_from_anatomical=kg.FrameTransform.identity(),
        anatomical_coordinate_system=ct.anatomical_coordinate_system
    )
    
    # Generate camera views
    # TODO offset views by hdf...
    ap_view, lateral_view = env.generate_views(annotation, lateral_translate=lateral_translation, superior_translate=superior_translation)
    
    env.device.source_to_detector_distance = file[f'device/source_to_detector_distance'][()].item()

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 10
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 2) # may increase for real-world tasks

    num_rollouts = 1
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
            heatmap_dict = dict()
            for cam_name in config['camera_names']:
                image_dict[cam_name] = file[f'/observations/images/{cam_name}'][starting_timestep]
                heatmap_dict[cam_name] = np.array(file[f'/observations/images/{cam_name}_heatmap'][()])
                print(np.shape(heatmap_dict[cam_name]))
            
            # TODO get new starting qpos from the sim environment directly
            print(f"starting qpos: {qpos_h5data[starting_timestep]}")
            qpos = torch.from_numpy(qpos_h5data[starting_timestep]).float().numpy()

            for t in range(max_timesteps):
                # new axis for different cameras
                all_cam_images = []
                for cam_name in config['camera_names']:
                    img = image_dict[cam_name].astype(np.float32)  # Ensure float32
                    heatmap = heatmap_dict[cam_name]  # Ensure float32
                    # Ensure heatmap has shape (H, W), add channel dim
                    if heatmap.ndim == 2:
                        heatmap = heatmap[..., None]  # (H, W, 1)
                    # Concatenate along channel axis (last axis)
                    img_with_heatmap = np.concatenate([img, heatmap], axis=-1)  # (H, W, C+1)
                    all_cam_images.append(img_with_heatmap)
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

                cannula_ap = target_qpos[:2]
                cannula_lateral = target_qpos[2:4]
                linear_ap = target_qpos[4:6]
                linear_lateral = target_qpos[6:8]
                direction = target_qpos[8:11]

                # direction = estimate_3d_direction(qpos[i][8:10], qpos[i][10:12], ap_proj, lat_proj)
                # print(direction)
                cannula_point = triangulate_point(cannula_ap, cannula_lateral, projection_matrices["ap"], projection_matrices["lateral"])
                linear_point = triangulate_point(linear_ap, linear_lateral, projection_matrices["ap"], projection_matrices["lateral"])

                ap_image, _ = env.render_tools(
                    tool_poses={
                        "cannula": (geo.point(cannula_point), geo.vector(direction), True),
                        "linear_drive": (geo.point(linear_point), geo.vector(direction), True),
                    },
                    view=ap_view,
                    rotation=0,
                )
                # print(np.shape(ap_image), np.min(ap_image), np.max(ap_image))
                ap_processed = image_utils.process_drr(ap_image, neglog=False, invert=False, clahe=False)
                # print(np.shape(ap_processed), np.min(ap_processed), np.max(ap_processed))
                ap_images.append(ap_processed)
                ap_projection = env.device.get_camera_projection()
                
                # Generate lateral image  
                lateral_image, _ = env.render_tools(
                    tool_poses={
                        "cannula": (geo.point(cannula_point), geo.vector(direction), True),
                        "linear_drive": (geo.point(linear_point), geo.vector(direction), True),
                    },
                    view=lateral_view,
                    rotation=0,
                )
                lateral_processed = image_utils.process_drr(lateral_image, neglog=False, invert=False)
                # print(np.shape(lateral_processed), np.min(lateral_processed), np.max(lateral_processed))
                lateral_images.append(lateral_processed)
                lateral_projection = env.device.get_camera_projection()

                image_dict['ap'] = ap_processed
                image_dict['lateral'] = lateral_processed

                # print(f"target qpos: {target_qpos[:3]}")
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
        
        output_file = filename.replace('.hdf5', '_eval.hdf5')
        
        output_file = os.path.join(config['ckpt_dir'], os.path.split(filename)[1])
        # Save new HDF5 file
        print(f"Saving to: {output_file}")
        
        with h5py.File(output_file, 'w') as new_f:
            # Copy all original attributes
            for key in file.attrs.keys():
                new_f.attrs[key] = file.attrs[key]
            
            # Mark as regenerated
            new_f.attrs['regenerated'] = True
            
            qpos = np.array(qpos_history)
            qvel = np.vstack(([0] * 11, np.diff(qpos, axis=0)))
            qvel = qvel.tolist()
            
            # Copy observations
            new_f.create_dataset("action", data=np.array(qpos, dtype=np.float32))
            obs_grp = new_f.create_group("observations")
            obs_grp.create_dataset("qpos", data=np.array(qpos, dtype=np.float32))
            obs_grp.create_dataset("qvel", data=np.array(qvel, dtype=np.float32))
            
            # Copy annotations
            anno_grp = new_f.create_group("annotations")
            anno_grp.create_dataset("start", data=np.array(annotation.startpoint.tolist()))
            anno_grp.create_dataset("end", data=np.array(annotation.endpoint.tolist()))
            anno_grp.create_dataset("world_from_anatomical", data=file['annotations/world_from_anatomical'][:])
            
            # Add projection matrices
            proj_grp = obs_grp.create_group("projection_matrices")
            proj_grp.create_dataset("ap", data=np.array(ap_projection))
            proj_grp.create_dataset("lateral", data=np.array(lateral_projection))
            
            # Add regenerated images
            imgs_grp = obs_grp.create_group("images")
            imgs_grp.create_dataset("ap", data=ap_images)
            imgs_grp.create_dataset("lateral", data=lateral_images)

            # TODO
            imgs_grp.create_dataset("ap_heatmap", data=generate_heatmap(
                annotation.startpoint.tolist(),
                annotation.endpoint.tolist(),
                np.array(ap_projection),
                ap_images[0].shape[:2],  # (H, W)
                np.array(ct.world_from_anatomical)
            ))

            imgs_grp.create_dataset("lateral_heatmap", data=generate_heatmap(
                annotation.startpoint.tolist(),
                annotation.endpoint.tolist(),
                np.array(lateral_projection),
                lateral_images[0].shape[:2],  # (H, W)
                np.array(ct.world_from_anatomical)
            ))

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
    use_wandb = config.get('use_wandb', False)
    resume_from_checkpoint = config.get('resume_from_checkpoint', None)

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)
    
    # Initialize training state
    start_epoch = 0
    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    
    # Load checkpoint if resuming
    if resume_from_checkpoint is not None:
        if os.path.exists(resume_from_checkpoint):
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            checkpoint = torch.load(resume_from_checkpoint)
            
            # Load model state
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint with training state
                policy.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                train_history = checkpoint.get('train_history', [])
                validation_history = checkpoint.get('validation_history', [])
                min_val_loss = checkpoint.get('min_val_loss', np.inf)
                best_ckpt_info = checkpoint.get('best_ckpt_info', None)
                print(f"Loaded full checkpoint from epoch {start_epoch-1}, min_val_loss: {min_val_loss}")
            else:
                # Just model weights
                policy.load_state_dict(checkpoint)
                print("Loaded model weights only (no training state)")
        else:
            print(f"Warning: Checkpoint file {resume_from_checkpoint} not found. Starting fresh training.")
    
    # Watch model with wandb
    if use_wandb:
        wandb.watch(policy, log="all", log_freq=100)
    
    for epoch in tqdm(range(start_epoch, num_epochs)):
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)
 
            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        
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
        
        # Compute epoch training summary
        epoch_summary_train = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary_train['loss']
        
        # Log to wandb
        if use_wandb:
            log_dict = {}
            # Log validation metrics with val_ prefix
            for k, v in epoch_summary.items():
                log_dict[f'val_{k}'] = v.item()
            # Log training metrics with train_ prefix  
            for k, v in epoch_summary_train.items():
                log_dict[f'train_{k}'] = v.item()
            
            log_dict['epoch'] = epoch
            log_dict['learning_rate'] = optimizer.param_groups[0]['lr']
            log_dict['best_val_loss'] = min_val_loss
            
            wandb.log(log_dict, step=epoch)

        if epoch % 250 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            # Save full checkpoint with training state
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_history': train_history,
                'validation_history': validation_history,
                'min_val_loss': min_val_loss,
                'best_ckpt_info': best_ckpt_info,
                'config': config
            }
            torch.save(checkpoint, ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed, use_wandb)
            
            # Log checkpoint as wandb artifact
            if use_wandb:
                artifact = wandb.Artifact(f"checkpoint_epoch_{epoch}", type="model")
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed, use_wandb)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed, use_wandb=False):
    print("plotting history")
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure(figsize=(10, 6))
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        
        epochs_train = np.linspace(0, num_epochs-1, len(train_history))
        epochs_val = np.linspace(0, num_epochs-1, len(validation_history))
        
        plt.plot(epochs_train, train_values, label='train', alpha=0.7)
        plt.plot(epochs_val, val_values, label='validation', alpha=0.7)
        plt.tight_layout()
        plt.legend()
        plt.title(f'{key} - Training History')
        plt.xlabel('Epoch')
        plt.ylabel(key)
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        # Log to wandb
        if use_wandb:
            wandb.log({f"plots/{key}_history": wandb.Image(plt)}, commit=False)
        
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
    parser.add_argument('--action_dim', action='store', type=int, help='Action Dimension', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    # for wandb logging
    parser.add_argument('--use_wandb', action='store', help='Enable wandb logging', required=False, default=True, type=bool)
    parser.add_argument('--wandb_project', action='store', type=str, help='wandb project name', default='vertebroplasty-imitation')
    parser.add_argument('--wandb_entity', action='store', type=str, help='wandb entity/username', default=None)
    
    # for resuming training
    parser.add_argument('--resume_from_checkpoint', action='store', type=str, help='path to checkpoint to resume training from', default=None)
    
    main(vars(parser.parse_args()))
