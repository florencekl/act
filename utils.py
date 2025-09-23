import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from xray_transforms.xray_transforms import build_augmentation, build_augmentation_val
from PIL import Image
import glob

import IPython
import time
from PIL import Image
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
        episode_files = episode_files[:int(num_episodes * len(episode_files))]  # Limit to num_episodes if specified
    return sorted(episode_files)  # Sort for reproducible order

def get_cropped(image, center_x, center_y):
    # Create cropped image (1/8th the size) centered on the highest value of the heatmap
    h, w = image.shape[:2]
    crop_w, crop_h = w // 2, h // 2
    
    # Compute crop corners ensuring they are within the image bounds
    start_x = max(center_x - crop_w // 2, 0)
    start_y = max(center_y - crop_h // 2, 0)
    end_x = min(start_x + crop_w, w)
    end_y = min(start_y + crop_h, h)
    
    cropped = image[start_y:end_y, start_x:end_x]

    pil_img = Image.fromarray(cropped)
    img = np.array(pil_img.resize((256, 256), Image.BILINEAR))

    return img

def resize(image, size=(256, 256)):
    if image.shape[:2] != size:
        pil_img = Image.fromarray(image)
        image = np.array(pil_img.resize(size, Image.BILINEAR))
    
    
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = np.array([image, image, image]).transpose(1, 2, 0) if len(image.shape) == 2 else np.repeat(image, 3, axis=2)

    return image

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_files, dataset_dir, camera_names, norm_stats, augmentation_func):
        super(EpisodicDataset).__init__()
        self.episode_files = episode_files  # List of filenames
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.augmentation_func = augmentation_func
        self.is_sim = None

        self.start_ts = None
        self.chunk_size = None
        
        # Cache for file handles to avoid repeated open/close
        self._file_cache = {}
        self._cache_size_limit = 50  # Keep max 50 files open
        self._access_order = []  # For LRU eviction
        
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_files)
    
    def _get_file_handle(self, episode_filename):
        """Get cached file handle with LRU eviction."""
        dataset_path = os.path.join(self.dataset_dir, episode_filename)
        
        if episode_filename in self._file_cache:
            # Move to end (most recently used)
            self._access_order.remove(episode_filename)
            self._access_order.append(episode_filename)
            return self._file_cache[episode_filename]
        
        # Open new file
        try:
            file_handle = h5py.File(dataset_path, 'r')
        except Exception as e:
            print(f"Error opening {dataset_path}: {e}")
            raise
        
        # Add to cache
        self._file_cache[episode_filename] = file_handle
        self._access_order.append(episode_filename)
        
        # Evict oldest if cache is full
        if len(self._file_cache) > self._cache_size_limit:
            oldest_file = self._access_order.pop(0)
            old_handle = self._file_cache.pop(oldest_file)
            old_handle.close()
        
        return file_handle
    
    def __del__(self):
        """Clean up file handles."""
        for handle in self._file_cache.values():
            if handle:
                try:
                    handle.close()
                except:
                    pass

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_filename = self.episode_files[index]
        
        # Use cached file handle instead of opening/closing every time
        root = self._get_file_handle(episode_filename)
        
        original_action_shape = root['/action'].shape
        episode_len = original_action_shape[0]

        if self.chunk_size is not None:
            start_ts_list = range(0, episode_len, self.chunk_size)
            # print([start_ts for start_ts in start_ts_list])
        else:
            if self.start_ts is None:
                start_ts_list = [np.random.choice(episode_len)]
            else:
                start_ts_list = [self.start_ts]

        ret_image_data = []
        ret_qpos_data = []
        ret_action_data = []
        ret_is_pad = []
        for start_ts in start_ts_list:
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            
            for cam_name in self.camera_names:
                # if "ap" == cam_name:
                #     img = root[f'/observations/images/{cam_name}'][start_ts]
                #     crop_center = tuple(root[f'/annotations/crop_center_ap'][:])
                #     cropped_img = get_cropped(img, crop_center[0], crop_center[1])
                #     image_dict[f"{cam_name}_cropped"] = cropped_img
                #     image_dict[cam_name] = img
                # elif "lateral" == cam_name:
                #     img = root[f'/observations/images/{cam_name}'][start_ts]
                #     crop_center = tuple(root[f'/annotations/crop_center_lateral'][:])
                #     cropped_img = get_cropped(img, crop_center[0], crop_center[1])
                #     image_dict[f"{cam_name}_cropped"] = cropped_img
                #     image_dict[cam_name] = img
                img = root[f'/observations/images/{cam_name}'][start_ts]
                image_dict[cam_name] = resize(img)
                # print(np.min(image_dict[cam_name]), np.max(image_dict[cam_name]), np.mean(image_dict[cam_name]), image_dict[cam_name].shape)
                # image_dict[cam_name] = self.augmentation_func(img)

            action = root['/action'][start_ts:]
            action_len = episode_len - start_ts
            qvel = root['/observations/qvel'][start_ts:]
            qvel_len = episode_len - start_ts

            padded_action = np.zeros(original_action_shape, dtype=np.float32)
            padded_action[:action_len] = action
            padded_qvel = np.zeros(original_action_shape, dtype=np.float32)
            padded_qvel[:qvel_len] = qvel
            is_pad = np.zeros(episode_len)
            is_pad[action_len:] = 1
            qvel = root['/observations/qvel'][start_ts]

            # new axis for different cameras
            all_cam_images = []
            
            for cam_name in self.camera_names:
                img = image_dict[cam_name].astype(np.float32)  # Ensure float32
                all_cam_images.append(img)

            all_cam_images = np.stack(all_cam_images, axis=0)

            # construct observations
            image_data = torch.from_numpy(all_cam_images).float()  # Explicitly convert to float32
            # print("Image data shape:", image_data.shape)
            qpos_data = torch.from_numpy(qpos).float()
            is_pad = torch.from_numpy(is_pad).bool()
            action_data = torch.from_numpy(padded_qvel).float()

            # channel last -> channel first
            image_data = torch.einsum('k h w c -> k c h w', image_data)

            action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

            ret_image_data.append(image_data)
            ret_qpos_data.append(qpos_data)
            ret_action_data.append(action_data)
            ret_is_pad.append(is_pad)

        # print(f"utils: chunk_size: {self.chunk_size} ret_image_data[0].shape: {ret_image_data[0].shape}, ret_qpos_data[0].shape: {ret_qpos_data[0].shape}, ret_action_data[0].shape: {ret_action_data[0].shape}, ret_is_pad[0].shape: {ret_is_pad[0].shape}")
        if self.chunk_size is not None:
            return ret_image_data, ret_qpos_data, ret_action_data, ret_is_pad
        else:
            return ret_image_data[0], ret_qpos_data[0], ret_action_data[0], ret_is_pad[0]
        # return ret_image_data[0], ret_qpos_data[0], ret_action_data[0], ret_is_pad[0]


def get_norm_stats(dataset_dir, episode_files):
    all_qpos_data = []
    all_action_data = []
    all_qvel_data = []
    for episode_filename in episode_files:
        dataset_path = os.path.join(dataset_dir, episode_filename)
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_qvel_data.append(torch.from_numpy(qvel))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_qvel_data = torch.stack(all_qvel_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    # normalize qvel data
    qvel_mean = all_qvel_data.mean(dim=[0, 1], keepdim=True)
    qvel_std = all_qvel_data.std(dim=[0, 1], keepdim=True)
    qvel_std = torch.clip(qvel_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": qvel_mean.numpy().squeeze(), "action_std": qvel_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, episodes_start, camera_names, batch_size_train, batch_size_val, train_dir=None, val_dir=None, episode_len=None, chunk_size=None):
    print(f'\nData from: {dataset_dir}\n')

    if train_dir is not None and val_dir is not None:
        # episode_files = discover_episode_files(train_dir, num_episodes, episodes_start)

        # val_files = discover_episode_files(val_dir, num_episodes, episodes_start)
        # episode_files = train_files + val_files
        
        train_files = discover_episode_files(train_dir, num_episodes, episodes_start)
        val_files = discover_episode_files(val_dir, num_episodes, episodes_start)
        episode_files = train_files + val_files

        # num_episodes = len(train_files) + len(val_files)
        num_episodes = len(episode_files)
        
        # TODO maybe change this again, but currently train-validation seems terrible and optimizing badly
        # obtain train test split
        # train_ratio = 0.8
        # shuffled_indices = np.random.permutation(num_episodes)
        # train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
        # val_indices = shuffled_indices[int(train_ratio * num_episodes):]
        
        # Split episode files based on indices
        # train_files = [episode_files[i] for i in train_indices]
        # val_files = [episode_files[i] for i in val_indices]

        # print(f'Found {len(train_files)} training files and {len(val_files)} validation files.')
        if len(train_files) == 0 or len(val_files) == 0:
            raise ValueError(f"No HDF5 files found in {train_dir} or {val_dir}")
        
        # obtain normalization stats for qpos and action
        norm_stats = get_norm_stats(train_dir, train_files)

        # construct dataset and dataloader
        train_dataset = EpisodicDataset(train_files, train_dir, camera_names, norm_stats, build_augmentation)
        if episode_len is not None and chunk_size is not None:
            datasets = []
            for start in range(0, episode_len, chunk_size):
                ds = EpisodicDataset(val_files, train_dir, camera_names, norm_stats, build_augmentation_val)
                ds.start_ts = start
                datasets.append(ds)
                print(ds.start_ts)
            print(datasets)
            val_dataset = ConcatDataset(datasets)
        else:
            val_dataset = EpisodicDataset(val_files, val_dir, camera_names, norm_stats, build_augmentation_val)
    else:
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
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=2, persistent_workers=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=2, persistent_workers=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=12, prefetch_factor=4, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=12, prefetch_factor=4, persistent_workers=True)

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

    