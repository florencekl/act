import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
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

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_files, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_files = episode_files  # List of filenames
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        
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
        # projection_matrices = {}
        # for cam_name in self.camera_names:
        #     projection_matrices[cam_name] = root[f'/observations/projection_matrices/{cam_name}'][()]

        target_size = (256, 256)  # or (64, 64) depending on your preference
        
        if not 'observations/heatmaps' in root:
            for cam_name in self.camera_names:
                # new approach TODO with cropped mask and 3 channel images
                img = root[f'/observations/images/{cam_name}'][start_ts]
            
                # Resize image if it doesn't match target size
                if img.shape[:2] != target_size:
                    pil_img = Image.fromarray(img)
                    img = np.array(pil_img.resize(target_size, Image.BILINEAR))

                # for normal backbone TODO
                # image_dict[cam_name] = np.array([img, img, img]).transpose(1, 2, 0)

                # for xrv backbone
                img = img[..., None]
                image_dict[cam_name] = img
        else:
            for cam_name in self.camera_names:
                # new approach TODO with cropped mask and 3 channel images using RGB pretrained weights again maybe?


                # current approach TODO
                images = root[f'/observations/images/{cam_name}'][start_ts]
                masks = root[f'/observations/masks/{cam_name}'][start_ts].astype(np.float32)
                masks = masks / 255.0  # Normalize masks to [0, 1]
                heatmap = np.array(root[f'/observations/heatmaps/{cam_name}'])
                image_dict[cam_name] = np.array([images, masks, heatmap]).transpose(1, 2, 0)


                # lateral_masks = np.array(masks['lateral'])
                # shape = root[f'/observations/masks/{cam_name}'].attrs["original_shape"]
                # masks = np.unpackbits(root[f'/observations/masks/{cam_name}'][:])[:np.prod(shape)].reshape(shape)[start_ts]
                # print(f"np.shape(images): {np.shape(images)}")
                # print(f"np.shape(image_dict[{cam_name}]): {np.shape(image_dict[cam_name])}")
                # print(f"np.min(image_dict[cam_name]): {np.min(image_dict[cam_name])}, np.max(image_dict[cam_name]): {np.max(image_dict[cam_name])}")
                # TODO create PIL rgb image from FULL DICTIONRAY ENTRY * 255 to uint8
                # from PIL import Image
                # img = Image.fromarray((image_dict[cam_name] * 255).astype(np.uint8), mode='RGB')
                # img.save(f'/data/flora/vertebroplasty_training/NMDID_v1_11_action_pretraining/{cam_name}_image.png')


                # heatmap_dict[cam_name] = root[f'/observations/images/{cam_name}_heatmap'][()]
        # get all actions after and including start_ts
        if is_sim:
            action = root['/action'][start_ts:]
            action_len = episode_len - start_ts
            qvel = root['/observations/qvel'][start_ts:]
            qvel_len = episode_len - start_ts
        else:
            action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
            action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned
            qvel = root['/observations/qvel'][max(0, start_ts - 1):]
            qvel_len = episode_len - max(0, start_ts - 1)

        self.is_sim = is_sim
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
            
            # TODO DEBUG if you want to visualize heatmaps and images that we use for training
            # from visualize_heatmaps import visualize_images_and_heatmaps
            # visualize_images_and_heatmaps(image_dict, heatmap_dict, 0, 0)
        all_cam_images = np.stack(all_cam_images, axis=0)
        # print(f"np.shape(all_cam_images): {np.shape(all_cam_images)}")

        # construct observations
        image_data = torch.from_numpy(all_cam_images).float()  # Explicitly convert to float32
        qpos_data = torch.from_numpy(qpos).float()
        # qvel_data = torch.from_numpy(qvel).float()
        # action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # qvel!
        # qpos_data = torch.from_numpy(qvel).float()
        qvel_data = torch.from_numpy(qvel).float()
        action_data = torch.from_numpy(padded_qvel).float()

        # channel last -> channel first
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        # Visualize the first image using PIL
        # Convert tensor to numpy array and change from channel-first (C, H, W) to channel-last (H, W, C)
        # img_np = image_data[0].permute(1, 2, 0).numpy()
        # Assuming the image data is in range [0,1], scale it to [0,255]
        # img_np = (img_np * 255).astype('uint8')
        # pil_img = Image.fromarray(img_np)
        # pil_img.save("/data_vertebroplasty/flora/vertebroplasty_training/NMDID_v1.5_3D_delta/getitem.png")

        # normalize image and change dtype to float
        # TODO maybe train on qvel data instead?
        # image_data = image_data / 255.0
        # image_data = image_data
        # print(np.shape(qvel_data), np.shape(qpos_data), np.shape(action_data), np.shape(image_data), np.shape(is_pad))
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        # qvel_data = (qvel_data - self.norm_stats["qvel_mean"]) / self.norm_stats["qvel_std"]

        return image_data, qpos_data, action_data, is_pad


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

    # stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
    #          "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
    #          "example_qpos": qpos}
    
    # qvel
    # stats = {"action_mean": qvel_mean.numpy().squeeze(), "action_std": qvel_std.numpy().squeeze(),
    #          "qpos_mean": qvel_mean.numpy().squeeze(), "qpos_std": qvel_std.numpy().squeeze(),
    #          "example_qpos": qpos}
    stats = {"action_mean": qvel_mean.numpy().squeeze(), "action_std": qvel_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, episodes_start, camera_names, batch_size_train, batch_size_val, train_dir=None, val_dir=None):
    print(f'\nData from: {dataset_dir}\n')

    if train_dir is not None and val_dir is not None:
        episode_files = discover_episode_files(train_dir, num_episodes, episodes_start)
        # val_files = discover_episode_files(val_dir, num_episodes, episodes_start)
        # episode_files = train_files + val_files

        # num_episodes = len(train_files) + len(val_files)
        num_episodes = len(episode_files)
        
        # TODO maybe change this again, but currently train-validation seems terrible and optimizing badly
        # obtain train test split
        train_ratio = 0.8
        shuffled_indices = np.random.permutation(num_episodes)
        train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
        val_indices = shuffled_indices[int(train_ratio * num_episodes):]
        
        # Split episode files based on indices
        train_files = [episode_files[i] for i in train_indices]
        val_files = [episode_files[i] for i in val_indices]

        print(f'Found {len(train_files)} training files and {len(val_files)} validation files.')
        if len(train_files) == 0 or len(val_files) == 0:
            raise ValueError(f"No HDF5 files found in {train_dir} or {val_dir}")
        
        # obtain normalization stats for qpos and action
        norm_stats = get_norm_stats(train_dir, train_files)

        # construct dataset and dataloader
        train_dataset = EpisodicDataset(train_files, train_dir, camera_names, norm_stats)
        val_dataset = EpisodicDataset(val_files, train_dir, camera_names, norm_stats)
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=2, prefetch_factor=2, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=2, prefetch_factor=2, persistent_workers=True)

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

    