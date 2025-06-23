import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import sys

from imitate_episodes import make_policy
from utils import set_seed

import h5py

# TODO eval_bc function in imitate_episodes.py

def generate_heatmap(start_point, end_point, projection_matrix, image_shape, sigma=10.0):
    """
    Generate a Gaussian heatmap from 3D start and end points projected to image space.
    
    Args:
        start_point: 3D coordinates of start point (x, y, z)
        end_point: 3D coordinates of end point (x, y, z)
        projection_matrix: 4x4 projection matrix for the camera
        image_shape: (height, width) of the target image
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        heatmap: 2D numpy array with Gaussian heatmap
    """
    height, width = image_shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Convert 3D points to homogeneous coordinates
    start_homo = np.array([start_point[0], start_point[1], start_point[2], 1.0])
    end_homo = np.array([end_point[0], end_point[1], end_point[2], 1.0])
    
    # Project points to image space
    start_proj = projection_matrix @ start_homo
    end_proj = projection_matrix @ end_homo
    
    # Convert from homogeneous to 2D coordinates
    if start_proj[2] > 0:  # Check if point is in front of camera
        start_2d = start_proj[:2] / start_proj[2]
        start_x, start_y = int(start_2d[0]), int(start_2d[1])
    else:
        start_x, start_y = -1, -1  # Invalid projection
        
    if end_proj[2] > 0:  # Check if point is in front of camera
        end_2d = end_proj[:2] / end_proj[2]
        end_x, end_y = int(end_2d[0]), int(end_2d[1])
    else:
        end_x, end_y = -1, -1  # Invalid projection
    
    # Generate Gaussian heatmaps for valid projections
    points_to_draw = []
    if 0 <= start_x < width and 0 <= start_y < height:
        points_to_draw.append((start_x, start_y))
    if 0 <= end_x < width and 0 <= end_y < height:
        points_to_draw.append((end_x, end_y))
        
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    
    for px, py in points_to_draw:
        # Calculate distance from each pixel to the point
        dist_sq = (x_grid - px) ** 2 + (y_grid - py) ** 2
        # Generate Gaussian
        gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
        # Add to heatmap (taking maximum to handle overlapping Gaussians)
        heatmap = np.maximum(heatmap, gaussian)
        
    return heatmap


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
    action_dim = args['action_dim'] if 'action_dim' in args else 14

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
                         'input_channels': 4  # RGB + heatmap
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names, 'input_channels': 4}  # RGB + heatmap
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

    ckpt_names = [f'policy_best.ckpt']
    results = []
    for ckpt_name in ckpt_names:
        success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
        results.append([ckpt_name, success_rate, avg_return])

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

    num_rollouts = 50
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

        with torch.inference_mode():
            for t in range(max_timesteps):
                
                file = h5py.File("/home/flora/projects/verteboplasty_imitation/external/data/sim_vertebroplasty_simple/episode_10.hdf5", 'r')

                original_action_shape = file['/action'].shape
                episode_len = original_action_shape[0]

                # get observation at start_ts only
                qpos = file['/observations/qpos'][0]
                qvel = file['/observations/qvel'][0]
                image_dict = dict()
                heatmap_dict = dict()
                
                # Load annotations and projection matrices
                annotation_start = file['/annotations/start'][()]
                annotation_end = file['/annotations/end'][()]
                projection_matrices = {}
                for cam_name in config['camera_names']:
                    projection_matrices[cam_name] = file[f'/observations/projection_matrices/{cam_name}'][()]
                
                for cam_name in config['camera_names']:
                    image_dict[cam_name] = file[f'/observations/images/{cam_name}'][0]
                    # Generate heatmap for this camera
                    heatmap_dict[cam_name] = generate_heatmap(
                        annotation_start, annotation_end, 
                        projection_matrices[cam_name], 
                        image_dict[cam_name].shape[:2]  # (H, W)
                    )
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
                    img = image_dict[cam_name].astype(np.float32)  # Ensure float32
                    heatmap = heatmap_dict[cam_name].astype(np.float32)  # Ensure float32
                    # Ensure heatmap has shape (H, W), add channel dim
                    if heatmap.ndim == 2:
                        heatmap = heatmap[..., None]  # (H, W, 1)
                    # Concatenate along channel axis (last axis)
                    img_with_heatmap = np.concatenate([img, heatmap], axis=-1)  # (H, W, C+1)
                    all_cam_images.append(img_with_heatmap)
                all_cam_images = np.stack(all_cam_images, axis=0)

                # construct observations
                image_data = torch.from_numpy(all_cam_images).float()  # Explicitly convert to float32
                qpos_data = torch.from_numpy(qpos).float()
                action_data = torch.from_numpy(padded_action).float()
                is_pad = torch.from_numpy(is_pad).bool()

                # channel last
                image_data = torch.einsum('k h w c -> k c h w', image_data)

                # normalize image and change dtype to float
                image_data = image_data / 255.0

                qpos = pre_process(qpos_data)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos

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

                print(f"targe_qpos: {target_qpos}")
                print(f"raw_action: {raw_action}")
                print(f"action: {action}")
                print(f"qpos: {qpos}")
                print(f"actual_cpos: {file['/observations/qpos'][1]}")

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

