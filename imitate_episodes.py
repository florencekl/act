from datetime import datetime
from logging import root
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv
import glob
from typing import Any, Tuple
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

from deepdrr_simulation_platform._generate_comparison_gif import triangulate_point
from deepdrr_simulation_platform import EpisodeData, calculate_distances, load_config, SimulationEnvironment
from deepdrr_simulation_platform.sim_environment import centroid_heatmap, centroid_with_bbox
from deepdrr_simulation_platform._data_validation import validate_trajectory_from_points, load_and_transform_mesh, validate_trajectory
from utils import load_data # data functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from deepdrr.vol import Mesh
from PIL import Image
from xray_transforms.xray_transforms import build_augmentation, build_augmentation_val, build_replay_augmentation_val, build_augmentation_real_xrays

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
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    action_dim = args['action_dim']
    
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
    if 'train_dir' in task_config:
        train_dir = task_config['train_dir']
        val_dir = task_config['val_dir']
        test_dir = task_config['test_dir']
    else:
        train_dir = None
        val_dir = None
        test_dir = None
    num_episodes = task_config['num_episodes']
    episodes_start = task_config['episode_start']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    # TODO change state_dim according to task
    state_dim = action_dim
    lr_backbone = 1e-5
    backbone = 'resnet18'
    # backbone = 'resnet34'
    # backbone = 'xrv_densenet121'
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
        'episodes_start': episodes_start,
        'action_dim': action_dim,
        'resume_from_checkpoint': args.get('resume_from_checkpoint', None),
        'warmup_epochs': args.get('warmup_epochs', 100),
        'lr_decay_type': args.get('lr_decay_type', 'cosine'),
        'pretrained_backbone_path': args.get('pretrained_backbone_path', None)
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        # ckpt_names = [f'policy_last.ckpt']
        # ckpt_names = [f'policy_epoch_1400_seed_0.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, dataset_dir=test_dir)
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
    
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, 
                                                           episodes_start, camera_names, 
                                                           batch_size_train, batch_size_val, 
                                                           train_dir=train_dir, val_dir=val_dir, 
                                                        #    episode_len=episode_len, chunk_size=policy_config['num_queries']
                                                           )
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


def load_pretrained_backbone_weights(policy, pretrained_path, camera_names):
    """
    Load pretrained backbone weights from another ACT model.
    
    Args:
        policy: Current ACT policy with separate backbones
        pretrained_path: Path to pretrained ACT model checkpoint
        camera_names: List of camera names for mapping
    """
    print(f"\n=== Loading Pretrained Backbone Weights ===")
    print(f"Source: {pretrained_path}")
    print(f"Target cameras: {camera_names}")
    
    # Load pretrained checkpoint
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")
    else:
        pretrained_state_dict = checkpoint
        print("Loaded model weights directly")
    
    # Find backbone-related keys in pretrained model
    backbone_keys = [k for k in pretrained_state_dict.keys() if 'backbone' in k]
    print(f"\nFound {len(backbone_keys)} backbone-related parameters in pretrained model:")
    
    # Group by backbone index
    backbone_groups = {}
    for key in backbone_keys:
        # Extract backbone index (e.g., "backbones.0.layer1.weight" -> 0)
        parts = key.split('.')
        if len(parts) >= 2 and parts[0] == 'backbones':
            backbone_idx = int(parts[1])
            if backbone_idx not in backbone_groups:
                backbone_groups[backbone_idx] = []
            backbone_groups[backbone_idx].append(key)
    
    for idx, keys in backbone_groups.items():
        cam_name = camera_names[idx] if idx < len(camera_names) else f"camera_{idx}"
        print(f"  Backbone {idx} ({cam_name}): {len(keys)} parameters")
        if len(keys) <= 5:  # Show first few parameter names if not too many
            for key in keys[:3]:
                print(f"    - {key}")
            if len(keys) > 3:
                print(f"    - ... and {len(keys)-3} more")
    
    # Load weights into current model
    current_state_dict = policy.state_dict()
    loaded_count = 0
    skipped_count = 0
    
    print(f"\nLoading weights into current model:")
    
    for backbone_idx in range(len(camera_names)):
        cam_name = camera_names[backbone_idx]
        
        if backbone_idx in backbone_groups:
            # Load weights for this backbone
            backbone_loaded = 0
            backbone_skipped = 0
            
            for pretrained_key in backbone_groups[backbone_idx]:
                if pretrained_key in current_state_dict:
                    # Check shape compatibility
                    if current_state_dict[pretrained_key].shape == pretrained_state_dict[pretrained_key].shape:
                        current_state_dict[pretrained_key] = pretrained_state_dict[pretrained_key]
                        backbone_loaded += 1
                        loaded_count += 1
                    else:
                        print(f"    Shape mismatch for {pretrained_key}: {current_state_dict[pretrained_key].shape} vs {pretrained_state_dict[pretrained_key].shape}")
                        backbone_skipped += 1
                        skipped_count += 1
                else:
                    backbone_skipped += 1
                    skipped_count += 1
            
            print(f"  Backbone {backbone_idx} ({cam_name}): loaded {backbone_loaded}, skipped {backbone_skipped}")
        else:
            print(f"  Backbone {backbone_idx} ({cam_name}): no pretrained weights found")
    
    # Load the updated state dict
    policy.load_state_dict(current_state_dict)
    
    print(f"\n=== Summary ===")
    print(f"Total parameters loaded: {loaded_count}")
    print(f"Total parameters skipped: {skipped_count}")
    print(f"Loading completed successfully!\n")


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

def initialize_environment_for_episode(episode: EpisodeData, ct=None) -> Tuple[SimulationEnvironment, Any, str, LineAnnotation, Any, Any, int]:
    """
    Initializes the simulation environment, loads phantom, tools, mesh, and annotation for a given episode.
    Returns: env, ct, tag, annotation, ap_view, lateral_view, rotation
    """
    cfg = load_config("/home/flora/projects/verteboplasty_imitation/deepdrr_simulation_platform/config.yaml")
    env = SimulationEnvironment(cfg)
    
    if ct is None:
        # Load phantom and tools
        if "blanca" in episode.phantoms_dir:
            phantom_path = os.path.join(episode.phantoms_dir, episode.case, 'TORSO')
            print(f"Loading phantom from: {phantom_path}")
            ct = env.load_phantom(phantom_path, from_density=True)
        else: 
            phantom_path = os.path.join(episode.phantoms_dir, episode.case)
            print(f"Loading phantom from: {phantom_path}")
            ct = env.load_phantom(phantom_path, from_density=False)
        print(f"Loading tools for case: {episode.case}")
    else:
        env.ct = ct
        env.right_in_world = ct.world_from_anatomical @ kg.vector(1, 0, 0)
        env.anterior_in_world = ct.world_from_anatomical @ kg.vector(0, 1, 0)
        env.superior_in_world = ct.world_from_anatomical @ kg.vector(0, 0, 1)
    tools = env.load_tools()
    
    tag = cfg.paths.vertebra_directory \
        + "/" + episode.case \
        + "/" + cfg.paths.vertebra_subfolder \
        + "/" + os.path.split(os.path.dirname(episode.annotation_path))[-1]

    mesh = Mesh.from_stl(tag + ".stl", material="bone", tag=tag, convert_to_RAS=True, world_from_anatomical=ct.world_from_anatomical)
    env.tools[tag] = mesh
    env.tools[tag].enabled = False
    env.initialize_projector()
    env.load_screws(6)
    env.device.source_to_detector_distance = int(episode.source_to_detector_distance[0])

    if "_R" in episode.annotation_path:
        rotation = 31 + 270
    elif "_L" in episode.annotation_path:
        rotation = 31 + 90
    else:
        rotation = 31
    rotation = int(rotation + 30)

    # Create annotation object
    # TODO depending on which dataset we are working with we have to apply this fix..
    annotation = LineAnnotation(
        # startpoint=ct.world_from_anatomical.inv @ geo.point(episode.start_point[:3]),
        # endpoint=ct.world_from_anatomical.inv @ geo.point(episode.end_point[:3]),
        startpoint=geo.point(episode.start_point[:3]),
        endpoint=geo.point(episode.end_point[:3]),
        volume=ct,
        world_from_anatomical=ct.world_from_anatomical,
        anatomical_coordinate_system=ct.anatomical_coordinate_system
    )

    # TODO this randomization is not working right now
    env.randomize_background(annotation.startpoint_in_world)

    annotation_original = episode.annotation_path
    annotation_opposite = annotation_original.replace("_L.fcsv", "_R.fcsv") if "_L.fcsv" in annotation_original else annotation_original.replace("_R.fcsv", "_L.fcsv")

    with open(annotation_original) as f:
        positions = []
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split(",")
            x, y, z = map(float, parts[1:4])
            # if cfg.paths.convert_to_RAS:
            positions.append([x * -1, y  * -1, z])
            # else:
            # positions.append([x, y, z])
        line_annotation_original = LineAnnotation(
                startpoint=geo.point(positions[0]),
                endpoint=geo.point(positions[1]), 
                volume=ct,
                world_from_anatomical=ct.world_from_anatomical,
                anatomical_coordinate_system=ct.anatomical_coordinate_system
                )
    with open(annotation_opposite) as f:
        positions = []
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split(",")
            x, y, z = map(float, parts[1:4])
            # if cfg.paths.convert_to_RAS:
            positions.append([x * -1, y  * -1, z])
            # else:
            # positions.append([x, y, z])
        line_annotation_opposite = LineAnnotation(
                startpoint=geo.point(positions[0]),
                endpoint=geo.point(positions[1]), 
                volume=ct,
                world_from_anatomical=ct.world_from_anatomical,
                anatomical_coordinate_system=ct.anatomical_coordinate_system
                )

    vec = line_annotation_original.direction_in_world + line_annotation_opposite.direction_in_world

    end1 = line_annotation_original.startpoint_in_world.lerp(line_annotation_original.endpoint_in_world, cfg.simulation.start_progress)
    end2 = line_annotation_opposite.startpoint_in_world.lerp(line_annotation_opposite.endpoint_in_world, cfg.simulation.start_progress)

    point = np.array(end1)[:3] + np.array(end2)[:3]
    direction_vector = vec / vec.norm()
    normalized_direction = direction_vector
    mid_point = point / 2

    # Generate camera views
    # print(episode.ap_direction)
    # print(episode.lateral_direction)
    # print(episode.ct_offset)
    ap_view, lateral_view = env.generate_views(
        annotation,
        ap_direction=geo.Vector3D(episode.ap_direction) if episode.ap_direction is not None else env.anterior_in_world,
        lateral_direction=geo.Vector3D(episode.lateral_direction) if episode.lateral_direction is not None else env.right_in_world,
        randomize=False
    )
    ap_view["point"] += geo.vector(list(episode.ct_offset)) if episode.ct_offset is not None else geo.vector([0,0,0])
    lateral_view["point"] += geo.vector(list(episode.ct_offset)) if episode.ct_offset is not None else geo.vector([0,0,0])

    # env.device.source_to_detector_distance = episode.source_to_detector_distance

    return env, ct, tag, annotation, ap_view, lateral_view, rotation, normalized_direction, mid_point

def crop_around_centroid(image: np.ndarray, centroid: tuple[int, int], bbox_size: tuple[int, int], 
                        crop_size: tuple[int, int] = None, padding_factor: float = 1.5) -> np.ndarray:
    """
    Crop image around centroid with perfect bounding box or custom size.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        centroid: (y, x) position of the centroid
        bbox_size: (height, width) of the object's bounding box
        crop_size: Optional (height, width) for fixed crop size. If None, uses bbox_size * padding_factor
        padding_factor: Multiplier for bbox_size when crop_size is None
    
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    center_y, center_x = centroid
    
    if crop_size is None:
        # Use bounding box size with padding
        crop_h = int(bbox_size[0] * padding_factor)
        crop_w = int(bbox_size[1] * padding_factor)
    else:
        crop_h, crop_w = crop_size

    # always make it square
    if crop_h > crop_w:
        crop_w = crop_h
    else:
        crop_h = crop_w
    
    # Calculate crop boundaries
    start_y = max(0, center_y - crop_h // 2)
    end_y = min(h, start_y + crop_h)
    start_x = max(0, center_x - crop_w // 2)
    end_x = min(w, start_x + crop_w)
    
    # Adjust start if end hits boundary
    if end_y == h:
        start_y = max(0, h - crop_h)
    if end_x == w:
        start_x = max(0, w - crop_w)
    
    return image[start_y:end_y, start_x:end_x]


def generate_random_vector(env: SimulationEnvironment) -> np.ndarray:
    while True:
        rand_vec = np.random.randn(3)
        proj = np.dot(rand_vec, env.anterior_in_world) * env.anterior_in_world
        perp_vec = rand_vec - proj
        norm = np.linalg.norm(perp_vec)
        if norm > 1e-6:
            return perp_vec / norm

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

def eval_bc(config, ckpt_name, save_episode=True, dataset_dir=None):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    max_timesteps = config['episode_len']
    max_timesteps = int(max_timesteps * 2) # may increase for real-world tasks
    temporal_agg = config['temporal_agg']

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    print(ckpt_path)
    print(policy_config)
    policy = make_policy(policy_class, policy_config)
    ckpt = torch.load(ckpt_path)
    # print(f"epoch: {ckpt.get('epoch', 'N/A')}, val_loss: {ckpt.get('val_loss', 'N/A')}")
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        loading_status = policy.load_state_dict(ckpt['model_state_dict'])
    else:
        loading_status = policy.load_state_dict(ckpt)
    print(loading_status)
    policy.cuda()
    policy.eval()
    # print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    # print(stats)

    files = [f"/data_vertebroplasty/flora/vertebroplasty_data/NMDID_subclustering_v1.2_2D_action+distance_8_vector/episode_{i}.hdf5" for i in range(120, 150)]
    if dataset_dir is not None:
        files = sorted(glob.glob(os.path.join(dataset_dir, '*.hdf5')))
    episode_previous = None
    ct = None
    regenerated_folder = os.path.join(config['ckpt_dir'], "regenerated_episodes")
    os.makedirs(regenerated_folder, exist_ok=True)
    csv_file = os.path.join(config['ckpt_dir'], "regenerated_episodes", "_episode_distances.csv")
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_number", "rollout_id", "distance"])
    for filename in files:
        ### --- SETUP SIMULATION ENVIRONMENT --- ###
        if filename is None:
            filename = "/data2/flora/vertebroplasty_imitation_1/episode_5001.hdf5"

        episode_original = EpisodeData.from_hdf5(filename)

        # if episode_original.episode <= 100:
        #     continue

        if episode_previous is not None and episode_previous.case == episode_original.case and ct is not None:
            env, ct, tag, annotation, ap_view, lateral_view, rotation, normalized_direction, mid_point = initialize_environment_for_episode(episode_original, ct=ct)
        else:
            del ct
            env, ct, tag, annotation, ap_view, lateral_view, rotation, normalized_direction, mid_point = initialize_environment_for_episode(episode_original)

        # env.device.source_to_detector_distance is already set in the function

        pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        query_frequency = policy_config['num_queries']
        # query_frequency = 10
        # query_frequency = 1
        if temporal_agg:
            # query_frequency = 5
            # query_frequency = 10
            query_frequency = 1
            num_queries = policy_config['num_queries']
    
        regenerated_episodes = []
        evaluation_distance_list = []

        num_rollouts = 1
        for rollout_id in range(num_rollouts):
            set_seed(rollout_id)
            lateral_images = []
            lateral_cropped = []
            lateral_masks = []
            lateral_heatmap= None
            ap_images = []
            ap_cropped = []
            ap_masks = []
            ap_heatmap= None
            qpos_history = []
            qvel_history = []

            # query_frequency = num_rollouts - rollout_id

            ### evaluation loop
            if temporal_agg:
                all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

            starting_timestep = 0

            target_size = (256, 256)  # or (64, 64) depending on your preference

            replay = None
            lambda_augs = None

            with torch.inference_mode():
                # get observation at start_ts only
                image_dict = dict()
                # heatmap_dict = dict()
                # for cam_name in config['camera_names']:
                #     if "ap" == cam_name:
                #         img = episode_original.images[cam_name][starting_timestep]
                #         crop_center = tuple(episode_original.crop_center_ap)
                #         cropped_img = get_cropped(img, crop_center[0], crop_center[1])
                #         image_dict[f"{cam_name}_cropped"] = cropped_img
                #         image_dict[cam_name] = img
                #     elif "lateral" == cam_name:
                #         img = episode_original.images[cam_name][starting_timestep]
                #         crop_center = tuple(episode_original.crop_center_lateral)
                #         cropped_img = get_cropped(img, crop_center[0], crop_center[1])
                #         image_dict[f"{cam_name}_cropped"] = cropped_img
                #         image_dict[cam_name] = img
                    # image_dict[cam_name] = build_augmentation_real_xrays(img, apply=True)
                # for cam_name in config['camera_names']:
                #     if "crop" in cam_name:
                #         image = np.array(Image.fromarray(episode_original.images[cam_name][starting_timestep]).resize(target_size, Image.BILINEAR))
                #     else:
                #         image = episode_original.images[cam_name][starting_timestep]
                #     image_dict[cam_name] = np.array([image, image, image]).transpose(1, 2, 0)
                    # episode_original.images[cam_name][starting_timestep]
                    # image_dict[cam_name] = build_augmentation_real_xrays(image_dict[cam_name])
                    # print(np.shape(episode_original.images[cam_name]))
                    # image_dict[cam_name] = build_augmentation_real_xrays(episode_original.images[cam_name][starting_timestep])


                # TODO get new starting qpos from the sim environment directly
                start_position = torch.from_numpy(episode_original.qpos[starting_timestep]).float().numpy()[:state_dim]

                if rollout_id != 0:

                    offset_direction = generate_random_vector(env)

                    offset_point = kg.point(
                        start_position[0] + np.random.uniform(env.cfg.simulation.cannula_offset[0] * 8, env.cfg.simulation.cannula_offset[1] * 4) * offset_direction[0],
                        start_position[1] + np.random.uniform(env.cfg.simulation.cannula_offset[0] * 8, env.cfg.simulation.cannula_offset[1] * 4) * offset_direction[1],
                        start_position[2] + np.random.uniform(env.cfg.simulation.cannula_offset[0] * 8, env.cfg.simulation.cannula_offset[1] * 4) * offset_direction[2],
                    )

                    print(f"Offsetting starting position by {offset_point - kg.point(*start_position[:3])} mm")

                    start_position[:3] = offset_point.tolist()[:3]

                # print(f"start position {start_position}")
                # if rollout_id == 0:
                start_position[:3] = mid_point[:3]
                # start_position[3:6] = -normalized_direction[:3]
                start_position[3:6] = -ap_view['direction'][:3]
                # start_position[3:6] = env.anterior_in_world
                # print(f"start position {start_position}")

                # ! qvel delta positioning
                qvel = torch.from_numpy(episode_original.qvel[starting_timestep]).float().numpy()[:state_dim]
                qvel_history.append(qvel)
                if state_dim == 11:
                    pedicle_side_flag = qvel[10]
                # print(qvel)
                action = np.zeros_like(qvel)

                xrays = {}
                crop_center = {}
                crop_bbox = {}

                env.ct.set_enabled(True)
                views = {
                    'ap_view': ap_view,
                    'lateral_view': lateral_view
                }
                for view_name, view in views.items():
                    env.tools.get("cannula").enabled = False
                    env.tools.get("linear_drive").enabled = False
                    env.tools.get(tag).enabled = False
                    env.projector.device.set_view(**view)
                    image = env.projector()
                    print(np.shape(image), np.min(image), np.max(image))
                    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                    xrays[view_name] = image
                    env.tools.get(tag).enabled = True
                    mask = centroid_heatmap(env.projector.project_seg(tags=[tag])[0])
                    
                    centroid, bbox_size = centroid_with_bbox(env.projector.project_seg(tags=[tag])[0])
                    max_idx =  np.unravel_index(np.argmax(mask), mask.shape)
                    centroid = max_idx
                    if 'ap' in view_name and episode_original.crop_center_ap is not None:
                        centroid = tuple(episode_original.crop_center_ap)
                    elif 'lateral' in view_name and episode_original.crop_center_lateral is not None:
                        centroid = tuple(episode_original.crop_center_lateral)
                    crop_center[view_name] = centroid
                    crop_bbox[view_name] = bbox_size
                env.ct.set_enabled(False)
                env.ct.set_enabled(True)
                
                goal_position = annotation.startpoint_in_world.lerp(annotation.endpoint_in_world, env.cfg.simulation.end_progress)

                with tqdm(range(max_timesteps), desc="Timesteps") as pbar:
                    for t in pbar:

                        base_point = start_position[:3]
                        direction = geo.vector(start_position[3:6])
                        distance = start_position[6]

                        cannula_point = base_point + (direction * distance)
                        linear_point = base_point + (direction * distance)

                        cannula_point_direction = geo.point(cannula_point) + geo.vector(direction).hat() * 10
                        linear_point_direction = geo.point(linear_point) + geo.vector(direction).hat() * 10

                        reached_goal_position = cannula_point

                        if t % query_frequency == 0:

                            ap_image, _ = env.render_tools(
                                tool_poses={
                                    "cannula": (geo.point(cannula_point), geo.vector(cannula_point_direction), True),
                                    "linear_drive": (geo.point(linear_point), geo.vector(linear_point_direction), False),
                                    tag: (None, None, False),
                                },
                                view=ap_view,
                                rotation=rotation,
                                generate_masks=False,
                            )
                            # ap_image = (ap_image - ap_image.min()) / (ap_image.max() - ap_image.min() + 1e-8)
                            # ap_image = 0.5 * xrays["ap_view"] + 0.5 * ap_image
                            # ap_image = env.histogram_matching(ap_image, xrays["ap_view"])

                            # ap_images.append(build_augmentation_real_xrays(ap_image))
                            ap_images.append(ap_image)
                            # ap_masks.append(masks["cannula"][0])
                            # ap_heatmap = (centroid_heatmap(masks[tag][0]))

                            ap_cropped.append(crop_around_centroid(ap_image, crop_center['ap_view'], crop_bbox['ap_view'], padding_factor=0.8))
                            # ap_cropped.append(get_cropped(ap_image, crop_center['ap_view'][1], crop_center['ap_view'][0]))
                            # Create cropped image (1/8th the size) centered on the highest value of the heatmap
                            # h, w = ap_image.shape[:2]
                            # crop_w, crop_h = w // 2, h // 2
                            
                            # # Find the position of the maximum value in the heatmap
                            # # max_idx = np.unravel_index(np.argmax(heatmap_lateral), heatmap_lateral.shape)
                            # center_y, center_x = crop_center['ap_view']

                            # # Compute crop corners ensuring they are within the image bounds
                            # start_x = max(center_x - crop_w // 2, 0)
                            # start_y = max(center_y - crop_h // 2, 0)
                            # end_x = min(start_x + crop_w, w)
                            # end_y = min(start_y + crop_h, h)

                            # # # ap_cropped.append(build_augmentation_real_xrays(ap_image[start_y:end_y, start_x:end_x]))
                            # ap_cropped.append(ap_image[start_y:end_y, start_x:end_x])

                            # # ap_images.append(SimulationEnvironment.process_image(ap_image, masks, ("cannula", tag), neglog=False, invert=False, clahe=False))
                            ap_projection = env.device.get_camera_projection()
                            
                            # Generate lateral image  
                            lateral_image, _ = env.render_tools(
                                tool_poses={
                                    "cannula": (geo.point(cannula_point), geo.vector(cannula_point_direction), True),
                                    "linear_drive": (geo.point(linear_point), geo.vector(linear_point_direction), False),
                                    tag: (None, None, False),
                                },
                                view=lateral_view,
                                rotation=rotation,
                                generate_masks=False,
                            )
                            # lateral_image = (lateral_image - lateral_image.min()) / (lateral_image.max() - lateral_image.min() + 1e-8)
                            # lateral_image = 0.5 * xrays["lateral_view"] + 0.5 * lateral_image
                            # lateral_image = env.histogram_matching(lateral_image, xrays["lateral_view"])

                            # lateral_images.append(build_augmentation_real_xrays(lateral_image))
                            lateral_images.append(lateral_image)
                            # lateral_masks.append(masks["cannula"][0])
                            # lateral_heatmap = (centroid_heatmap(masks[tag][0]))

                            
                            lateral_cropped.append(crop_around_centroid(lateral_image, crop_center['lateral_view'], crop_bbox['lateral_view'], padding_factor=0.8))
                            # lateral_cropped.append(get_cropped(lateral_image, crop_center['lateral_view'][1], crop_center['lateral_view'][0]))
                            

                            # Create cropped image (1/8th the size) centered on the highest value of the heatmap
                            # h, w = lateral_image.shape[:2]
                            # crop_w, crop_h = w // 2, h // 2
                            
                            # # Find the position of the maximum value in the heatmap
                            # # max_idx = np.unravel_index(np.argmax(heatmap_lateral), heatmap_lateral.shape)
                            # center_y, center_x = crop_center['lateral_view']

                            # # Compute crop corners ensuring they are within the image bounds
                            # start_x = max(center_x - crop_w // 2, 0)
                            # start_y = max(center_y - crop_h // 2, 0)
                            # end_x = min(start_x + crop_w, w)
                            # end_y = min(start_y + crop_h, h)

                            # # # lateral_cropped.append(build_augmentation_real_xrays(lateral_image[start_y:end_y, start_x:end_x]))
                            # lateral_cropped.append(lateral_image[start_y:end_y, start_x:end_x])

                            lateral_projection = env.device.get_camera_projection()
                            
                            # image_dict['ap'] = build_replay_augmentation_val(ap_images[-1], replay=replay, lambda_transforms=lambda_augs, apply=True)[0]
                            # image_dict['lateral'] = build_replay_augmentation_val(lateral_images[-1], replay=replay, lambda_transforms=lambda_augs, apply=True)[0]
                            # image_dict['ap'] = build_augmentation_real_xrays(ap_images[-1])
                            # image_dict['lateral'] = build_augmentation_real_xrays(lateral_images[-1])
                            image_dict['ap'] = np.array([ap_images[-1], ap_images[-1], ap_images[-1]]).transpose(1, 2, 0)
                            image_dict['lateral'] = np.array([lateral_images[-1], lateral_images[-1], lateral_images[-1]]).transpose(1, 2, 0)

                            crop = np.array(Image.fromarray(ap_cropped[-1]).resize(target_size, Image.BILINEAR))
                            image_dict['ap_cropped'] = np.array([crop, crop, crop]).transpose(1, 2, 0)
                            crop = np.array(Image.fromarray(lateral_cropped[-1]).resize(target_size, Image.BILINEAR))
                            image_dict['lateral_cropped'] = np.array([crop, crop, crop]).transpose(1, 2, 0)

                            # image_dict['ap_cropped'] = build_augmentation_real_xrays(image_dict['ap_cropped'])
                            # image_dict['lateral_cropped'] = build_augmentation_real_xrays(image_dict['lateral_cropped'])
                            # image_dict['ap'] = build_augmentation_real_xrays(image_dict['ap'])
                            # image_dict['lateral'] = build_augmentation_real_xrays(image_dict['lateral'])

                        # new axis for different cameras
                        all_cam_images = []
                        for cam_name in config['camera_names']:
                            img = image_dict[cam_name].astype(np.float32)  # Ensure float32
                            all_cam_images.append(img)
                        all_cam_images = np.stack(all_cam_images, axis=0)

                        # construct observations
                        image_data = torch.from_numpy(all_cam_images)

                        # channel last
                        image_data = torch.einsum('k h w c -> k c h w', image_data)
                        image_data = image_data.float().cuda().unsqueeze(0)  # add batch dimension
                        
                        # qpos_history.append(qvel)
                        qpos = pre_process(qvel)
                        qpos = torch.from_numpy(qvel).float().cuda().unsqueeze(0)

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
                        qvel_history.append(action)

                        action[7::] = np.round(action[7::])

                        # ! qvel delta relative position
                        start_position = start_position + action
                        qvel = action

                        qpos_history.append(start_position)
                        qvel_history.append(qvel)

                        cur_distance = np.linalg.norm(geo.point(goal_position) - geo.point(reached_goal_position))
                        
                        # Determine phase from first 3 one-hot flags and side from the last flag.
                        flags_phase = action[7:10]
                        phase_options = ['nav', 'ori', 'ins']
                        phase_index = int(np.argmax(flags_phase))
                        phase_str = phase_options[phase_index]
                        
                        if state_dim == 11:
                            action[10] = pedicle_side_flag
                            flag_side = action[10]
                            side_str = 'L' if round(flag_side) == 0 else 'R'
                        else:
                            side_str = 'N/A'

                        pbar.set_postfix({"distance": cur_distance, "phase": phase_str, "side": side_str})

            # Convert to numpy arrays
            # ap_images = np.array(ap_images[::5])
            # lateral_images = np.array(lateral_images[::5])
            ap_images = np.array(ap_images)
            lateral_images = np.array(lateral_images)

            # qvel = np.vstack(([0] * config['action_dim'], np.diff(qpos_history, axis=0)))
            # qvel = qvel.tolist()
            
            regenerated_episodes.append(EpisodeData.create_episode_data(
                case=episode_original.case,
                phantoms_dir=episode_original.phantoms_dir,
                anatomical_coordinate_system=episode_original.anatomical_coordinate_system,
                world_from_anatomical=episode_original.world_from_anatomical,
                annotation_path=episode_original.annotation_path,
                line_annotation=annotation,
                episode_number=episode_original.episode,
                ap_direction=ap_view['direction'],
                lateral_direction=lateral_view['direction'],
                source_to_detector_distance=env.device.source_to_detector_distance,
                ap_translations=episode_original.ap_translations,
                lateral_translations=episode_original.lateral_translations,
                projector_noise=episode_original.noise,
                photon_count=episode_original.photon_count,
                qpos=np.array(qpos_history, dtype=np.float32),
                qvel=np.array(qvel_history, dtype=np.float32),
                ap_projection=ap_projection,
                ct_offset=geo.vector(episode_original.ct_offset),
                lateral_projection=lateral_projection,
                ap_images=np.array(ap_images),
                lateral_images=np.array(lateral_images),
                # lateral_masks=np.array(lateral_masks, dtype=np.uint8),
                # ap_masks=np.array(ap_masks, dtype=np.uint8),
                # ap_heatmap=np.array(ap_heatmap),
                # lateral_heatmap=np.array(lateral_heatmap),
                lateral_masks=None,
                ap_masks=None,
                ap_heatmap=None,
                lateral_heatmap=None,
                ap_cropped=ap_cropped,
                lateral_cropped=lateral_cropped,
                ap_cropped_small=None,
                lateral_cropped_small=None,
            ))

            # tag = env.cfg.paths.vertebra_directory \
            #     + "/" + episode_original.case \
            #     + "/" + env.cfg.paths.vertebra_subfolder \
            #     + "/" + os.path.split(os.path.dirname(episode_original.annotation_path))[-1] + ".stl"

            # _, _, _, _, _, _, distance = calculate_distances(episode_original, regenerated_episodes[-1])
            # calculate difference between 2 positions
            goal_position = annotation.startpoint_in_world.lerp(annotation.endpoint_in_world, env.cfg.simulation.end_progress)

            evaluation_distance = np.linalg.norm(geo.point(goal_position) - geo.point(reached_goal_position))

            evaluation_distance_list.append(evaluation_distance)
            csv_file = os.path.join(config['ckpt_dir'], "regenerated_episodes", "_episode_distances.csv")
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([episode_original.episode, rollout_id, evaluation_distance])
            
            # # Load and transform mesh
            # mesh = load_and_transform_mesh(tag, episode_original.world_from_anatomical)
            
            # # Validate using existing function
            # valid_trajectory = validate_trajectory(qpos, mesh, 1, 0.25, 1)["is_valid"]
            
            # if valid_trajectory:
            #     break

        print(f"Timestamp: {datetime.now()} Distances: {evaluation_distance_list} argmin: {np.argmin(evaluation_distance_list)} -> distance {evaluation_distance_list[np.argmin(evaluation_distance_list)]}")

        regenerated_folder = os.path.join(config['ckpt_dir'], "regenerated_episodes")
        os.makedirs(regenerated_folder, exist_ok=True)
        env.save_episode(regenerated_episodes[np.argmin(evaluation_distance_list)], regenerated_folder, f"{episode_original.episode:04d}")

        episode_previous = episode_original

        del env, tag, annotation, ap_view, lateral_view, rotation
        

    return 0, 0


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    use_wandb = config.get('use_wandb', False)
    resume_from_checkpoint = config.get('resume_from_checkpoint', None)

    # val_dataloader.dataset.chunk_size = policy_config['num_queries']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    
    # Load pretrained backbone weights if specified
    if config.get('pretrained_backbone_path') is not None:
        load_pretrained_backbone_weights(policy, config['pretrained_backbone_path'], config['camera_names'])
    
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)
    
    # Learning rate scheduling setup
    # warmup_epochs = config.get('warmup_epochs', 100)  # Default warmup for 100 epochs
    # lr_decay_type = config.get('lr_decay_type', 'cosine')  # 'cosine', 'step', or 'none'
    # base_lr = policy_config['lr']
    
    # # Create learning rate scheduler with warmup + decay
    # def lr_lambda(current_epoch):
    #     if current_epoch < warmup_epochs:
    #         # Linear warmup: gradually increase from 0 to 1
    #         return current_epoch / warmup_epochs
    #     else:
    #         if lr_decay_type == 'cosine':
    #             # Cosine annealing after warmup
    #             import math
    #             progress = (current_epoch - warmup_epochs) / (num_epochs - warmup_epochs)
    #             return 0.5 * (1 + math.cos(math.pi * progress))
    #         elif lr_decay_type == 'step':
    #             # Step decay: reduce LR by factor of 0.1 every 1000 epochs after warmup
    #             steps = (current_epoch - warmup_epochs) // 1000
    #             return 0.1 ** steps
    #         else:  # lr_decay_type == 'none'
    #             # No decay after warmup, maintain constant LR
    #             return 1.0
    
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
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
                # start_epoch = checkpoint.get('epoch', 0) + 1
                # train_history = checkpoint.get('train_history', [])
                # print(train_history)
                validation_history = checkpoint.get('validation_history', [])
                min_val_loss = checkpoint.get('min_val_loss', np.inf)
                best_ckpt_info = checkpoint.get('best_ckpt_info', None)
                
                # # Load scheduler state if available
                # if 'scheduler_state_dict' in checkpoint:
                #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # else:
                #     # If no scheduler state, manually set the scheduler to the correct epoch
                #     for _ in range(start_epoch):
                #         scheduler.step()
                
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
    
    # Initialize progress bar
    pbar = tqdm(range(start_epoch, num_epochs), desc="Training")
    
    for epoch in pbar:
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            # for start_ts in range(0, config['episode_len'], policy_config['num_queries']):
            #     val_dataloader.dataset.start_ts = start_ts
            for batch_idx, data in enumerate(val_dataloader):
                # image_list, qpos_list, actions_list, is_pad_list = data
                # # timestep_dicts = []
                # for qpos, image, actions, is_pad in zip(qpos_list, image_list, actions_list, is_pad_list):
                #     # print(f"validation: qpos.shape: {qpos.shape}, image.shape: {image.shape}, actions.shape: {actions.shape}, is_pad.shape: {is_pad.shape}")
                #     epoch_dicts.append(forward_pass((image, qpos, actions, is_pad), policy))
                # forward_dict = compute_dict_mean(timestep_dicts)
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
        # for start_ts in range(0, config['episode_len'], policy_config['num_queries']):
        #         train_dataloader.dataset.start_ts = start_ts + np.random.choice(config['episode_len'])
        #         for batch_idx, data in enumerate(train_dataloader):
        #             forward_dict = forward_pass(data, policy)
        #             # backward
        #             loss = forward_dict['loss']
        #             loss.backward()
        #             optimizer.step()
        #             optimizer.zero_grad()
        #             train_history.append(detach_dict(forward_dict))

        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        
        # # Step the learning rate scheduler
        # scheduler.step()
        # current_lr = scheduler.get_last_lr()[0]
        
        # Compute epoch training summary
        epoch_summary_train = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary_train['loss']
        
        # Update progress bar with current losses
        pbar.set_postfix({
            'train_loss': f'{epoch_train_loss:.4f}',
            'val_loss': f'{epoch_val_loss:.4f}',
            'best_val': f'{min_val_loss:.4f}'
        })
        
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
            # log_dict['learning_rate'] = current_lr
            log_dict['best_val_loss'] = min_val_loss
            
            wandb.log(log_dict, step=epoch)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            # Save full checkpoint with training state
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                'train_history': train_history,
                'validation_history': validation_history,
                'min_val_loss': min_val_loss,
                'best_ckpt_info': best_ckpt_info,
                'config': config
            }
            torch.save(checkpoint, ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed, use_wandb)
            
            # Log checkpoint as wandb artifact
            # if use_wandb:
                # artifact = wandb.Artifact(f"checkpoint_epoch_{epoch}", type="model")
                # artifact.add_file(ckpt_path)
                # wandb.log_artifact(artifact)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # Save full checkpoint with training state
    ckpt_path = os.path.join(ckpt_dir, f'policy_best_full_state.ckpt')
    checkpoint = {
        'epoch': best_epoch,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict(),
        'train_history': train_history,
        'validation_history': validation_history,
        'min_val_loss': min_val_loss,
        'best_ckpt_info': best_ckpt_info,
        'config': config
    }
    torch.save(checkpoint, ckpt_path)

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
    parser.add_argument('--action_dim', action='store', type=int, help='Action Dimension', required=True)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=True)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=True)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=True)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=True)
    parser.add_argument('--temporal_agg', action='store_true')
    
    # for wandb logging
    parser.add_argument('--use_wandb', action='store', help='Enable wandb logging', required=False, default=True, type=bool)
    parser.add_argument('--wandb_project', action='store', type=str, help='wandb project name', default='vertebroplasty-imitation')
    parser.add_argument('--wandb_entity', action='store', type=str, help='wandb entity/username', default=None)
    
    # for resuming training
    parser.add_argument('--resume_from_checkpoint', action='store', type=str, help='path to checkpoint to resume training from', default=None)
    
    # for learning rate scheduling
    parser.add_argument('--warmup_epochs', action='store', type=int, help='number of epochs for learning rate warmup', default=0)
    parser.add_argument('--lr_decay_type', action='store', type=str, help='learning rate decay type: cosine, step, or none', default='none', choices=['cosine', 'step', 'none'])
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    
    # for loading pretrained backbone weights
    parser.add_argument('--pretrained_backbone_path', action='store', type=str, help='path to pretrained ACT model to load backbone weights from', default=None)
    
    main(vars(parser.parse_args()))
