import pathlib

### Task parameters
DATA_DIR = '/data2/flora/vertebroplasty_data'
ALT_DATA_DIR = '/data_vertebroplasty/flora/vertebroplasty_data'
SIM_TASK_CONFIGS = {
    'sim_vertebroplasty_simple': {
        'dataset_dir': DATA_DIR + '/vertebroplasty_imitation_custom_channels_xray_mask_heatmap_fixed',
        'episode_start': 0,
        'num_episodes': 4000,
        'episode_len': 100,
        'camera_names': ['ap', 'lateral']
    },
    'sim_vertebroplasty_full_2D': {
        'dataset_dir': DATA_DIR + '/custom_channels_projector_noise_scatter_action_12',
        'episode_start': 0,
        'num_episodes': 4000,
        'episode_len': 100,
        'camera_names': ['ap', 'lateral']
    },
    'sim_vertebroplasty_11_action_noise_screw_no_robot': {
        'dataset_dir': DATA_DIR + '/custom_channels_projector_scatter_action_11',
        'episode_start': 0,
        'num_episodes': 4000,
        'episode_len': 100,
        'camera_names': ['ap', 'lateral']
    },
    'sim_vertebroplasty_11_no_robot': {
        'dataset_dir': DATA_DIR + '/action_11_neglog_xray_only_noise',
        'episode_start': 0,
        'num_episodes': 100,
        'episode_len': 100,
        'camera_names': ['ap', 'lateral']
    },
    'NMDID_v1_11_action_pretraining': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1',
        'episode_start': 0,
        'num_episodes': 200,
        'episode_len': 100,
        'camera_names': ['ap', 'lateral']
    },
    'NMDID_v1_11_action_training': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1',
        'episode_start': 0,
        'num_episodes': 1000,
        'episode_len': 100,
        'camera_names': ['ap', 'lateral']
    },
    'NMDID_v1.1_9_action_training': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.1_2D_action+distance',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 100,
        'camera_names': ['ap', 'lateral'],
        'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.1_2D_action+distance/TRAIN',
        'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.1_2D_action+distance/VAL',
        'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.1_2D_action+distance/TEST',
    },
    'NMDID_v1.2_8_action_training': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.2_2D_action+distance_8_vector',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 100,
        'camera_names': ['ap', 'lateral'],
    },
    'NMDID_v1.3_7_action_training': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.3_3D_action+distance_7_vector',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 100,
        'camera_names': ['ap', 'lateral'],
        'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.3_3D_action+distance_7_vector/TRAIN',
        'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.3_3D_action+distance_7_vector/VAL',
        'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.3_3D_action+distance_7_vector/TEST',
    },
    'NMDID_v1.4_9_action_training': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.4_2D_action+distance_9_vector',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 50,
        'camera_names': ['ap', 'lateral'],
        'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.4_2D_action+distance_9_vector/TRAIN',
        'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.4_2D_action+distance_9_vector/VAL',
        'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.4_2D_action+distance_9_vector/TEST',
    },
    # 'NMDID_v1.5_3D_basics': { FAKE1.5
    #     'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.4_2D_action+distance_9_vector_with_base',
    #     'episode_start': 0,
    #     'num_episodes': 1.0,
    #     'episode_len': 50,
    #     'camera_names': ['ap', 'lateral'],
    #     'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.4_2D_action+distance_9_vector_with_base',
    #     'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.4_2D_action+distance_9_vector_with_base/VAL',
    #     'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.4_2D_action+distance_9_vector_with_base/TEST',
    # },
    'NMDID_v1.5_3D_basics': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.5_3D_9_basics',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 100,
        'camera_names': ['ap', 'lateral'],
        'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.5_3D_9_basics',
        'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.5_3D_9_basics/VAL',
        'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.5_3D_9_basics/TEST',
    },
    'NMDID_v1.6': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.6/valid_episodes',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 100,
        'camera_names': ['ap', 'lateral'],
        'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.6/valid_episodes',
        'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.6/valid_episodes',
        'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.6/valid_episodes',
    },
    'NMDID_v1.7': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.7/valid_episodes',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 100,
        'camera_names': ['ap', 'lateral'],
        'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.7/valid_episodes',
        'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.7/valid_episodes',
        'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.7/valid_episodes',
    },
    'NMDID_v1.8': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.8/valid_episodes',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 100,
        'camera_names': ['ap', 'lateral'],
        'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.8/valid_episodes',
        'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.8/valid_episodes',
        'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.8/TEST',
    },
    'NMDID_v1.9_cropped': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.9_cropped/valid_episodes',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 200,
        'camera_names': ['ap', 'lateral', 'ap_cropped', 'lateral_cropped'],
        'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.9_cropped/valid_episodes',
        'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.9_cropped/valid_episodes',
        # 'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.9_cropped/TEST',
        # 'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.9_cropped/TEST',
        'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.9_cropped/TEST',
    },
    'NMDID_v2.0': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.0',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 200,
        'camera_names': ['ap', 'lateral', 'ap_cropped', 'lateral_cropped'],
        'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.0',
        'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.0',
        'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.0/TEST',
    },
    'NMDID_v2.1': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.1_rand_dir',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 200,
        'camera_names': ['ap', 'lateral', 'ap_cropped', 'lateral_cropped'],
        'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.1_rand_dir/TRAINING',
        'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.1_rand_dir/VALIDATION',
        'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.1_rand_dir/TEST',
    },
    't10_left': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.9_cropped/valid_episodes/t10_left',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 200,
        'camera_names': ['ap', 'lateral', 'ap_cropped', 'lateral_cropped'],
        'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.9_cropped/valid_episodes/t10_left',
        'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.9_cropped/valid_episodes/t10_left',
        'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v1.9_cropped/valid_episodes/t10_left',
    },
    'NMDID_v2.2': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.2',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 200,
        'camera_names': ['ap', 'lateral', 'ap_cropped', 'lateral_cropped'],
        'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.2',
        'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.2',
        'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.1_rand_dir/TEST',
    },
    'NMDID_v2.4': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.4',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 200,
        'camera_names': ['ap', 'lateral', 'ap_cropped', 'lateral_cropped'],
        'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.4/TRAINING',
        'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.4/VALIDATION',
        'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.4/TEST',
    },
    'NMDID_v2.4_T11': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.4',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 200,
        'camera_names': ['ap', 'lateral', 'ap_cropped', 'lateral_cropped'],
        'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.4/TRAINING/T11',
        'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.4/VALIDATION/T11',
        'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.4/TEST/T11',
    },
    
    'NMDID_v2.5': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.5_quick',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 100,
        'camera_names': ['ap', 'lateral', 'ap_cropped', 'lateral_cropped'],
        'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.5_quick/TRAINING',
        'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.5_quick/VALIDATION',
        'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.5_quick/TEST',
    },

    'NMDID_v2.6': {
        'dataset_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.5_offset_rotation',
        'episode_start': 0,
        'num_episodes': 1.0,
        'episode_len': 100,
        'camera_names': ['ap', 'lateral', 'ap_cropped', 'lateral_cropped'],
        'train_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.5_offset_rotation/TRAINING',
        'val_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.5_offset_rotation/VALIDATION',
        'test_dir': ALT_DATA_DIR + '/NMDID_subclustering_v2.5_offset_rotation/TEST',
    },
}

### Simulation envs fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
