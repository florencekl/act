import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import glob

import IPython
import time
e = IPython.embed

def discover_episode_files(dataset_dir, num_episodes=None, episodes_start=None):
    """Discover all HDF5 files in the dataset directory."""
    pattern = os.path.join(dataset_dir, "*.hdf5")
    hdf5_files = glob.glob(pattern)
    # Return just the filenames without the full path
    episode_files = [os.path.basename(f) for f in hdf5_files]
    if episodes_start is not None:
        episode_files = episode_files[episodes_start:]  # Limit to episodes_start if specified
    if num_episodes is not None:
        episode_files = episode_files[:num_episodes]  # Limit to num_episodes if specified
    return sorted(episode_files)  # Sort for reproducible order

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_files, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_files = episode_files  # List of filenames
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_files)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_filename = self.episode_files[index]
        dataset_path = os.path.join(self.dataset_dir, episode_filename)
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)

            world_from_anatomical = root['annotations/world_from_anatomical'][()]
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            # heatmap_dict = dict()
            
            # Load annotations and projection matrices
            annotation_start = root['/annotations/start'][()]
            annotation_end = root['/annotations/end'][()]
            projection_matrices = {}
            for cam_name in self.camera_names:
                projection_matrices[cam_name] = root[f'/observations/projection_matrices/{cam_name}'][()]
            
            for cam_name in self.camera_names:
                # TODO
                images = root[f'/observations/images/{cam_name}'][start_ts]
                shape = root[f'/observations/masks/{cam_name}'].attrs["original_shape"]
                masks = np.unpackbits(root[f'/observations/masks/{cam_name}'][:])[:np.prod(shape)].reshape(shape)[start_ts]
                heatmap = np.array(root[f'/observations/heatmaps/{cam_name}'])
                image_dict[cam_name] = np.array([images, masks, heatmap]).transpose(1, 2, 0)
                # TODO create PIL rgb image from FULL DICTIONRAY ENTRY * 255 to uint8
                # from PIL import Image
                # img = Image.fromarray((image_dict[cam_name] * 255).astype(np.uint8), mode='RGB')
                # img.save(f'/data/flora/vertebroplasty_training/NMDID_v1_11_action_pretraining/{cam_name}_image.png')


                # heatmap_dict[cam_name] = root[f'/observations/images/{cam_name}_heatmap'][()]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            img = image_dict[cam_name].astype(np.float32)  # Ensure float32
            # heatmap = heatmap_dict[cam_name]  # Ensure float32
            # Ensure heatmap has shape (H, W), add channel dim
            # if heatmap.ndim == 2:
            #     heatmap = heatmap[..., None]  # (H, W, 1)
            # Concatenate along channel axis (last axis)
            # img_with_heatmap = np.concatenate([img, heatmap], axis=-1)  # (H, W, C+1)
            # all_cam_images.append(img_with_heatmap)
            all_cam_images.append(img)
            
            # TODO DEBUG if you want to visualize heatmaps and images that we use for training
            # from visualize_heatmaps import visualize_images_and_heatmaps
            # visualize_images_and_heatmaps(image_dict, heatmap_dict, 0, 0)
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images).float()  # Explicitly convert to float32
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last -> channel first
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, episode_files):
    all_qpos_data = []
    all_action_data = []
    for episode_filename in episode_files:
        dataset_path = os.path.join(dataset_dir, episode_filename)
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, episodes_start, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    
    # Discover all HDF5 files in the dataset directory
    episode_files = discover_episode_files(dataset_dir, num_episodes, episodes_start)
    num_episodes = len(episode_files)
    print(f'Found {num_episodes} episode files: {episode_files[:5]}{"..." if num_episodes > 5 else ""}')
    
    if num_episodes == 0:
        raise ValueError(f"No HDF5 files found in {dataset_dir}")
    
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]
    
    # Split episode files based on indices
    train_files = [episode_files[i] for i in train_indices]
    val_files = [episode_files[i] for i in val_indices]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, episode_files)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_files, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_files, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=2, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=2, persistent_workers=True)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    # print(epoch_dicts)
    # print(epoch_dicts[0])
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    