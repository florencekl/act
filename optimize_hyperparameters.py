import os
import torch
import optuna
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback

from policy import ACTPolicy
from utils import load_data, compute_dict_mean
from constants import SIM_TASK_CONFIGS

class ACTLightningModule(pl.LightningModule):
    def __init__(self, 
                 kl_weight: float = 10.0, 
                 chunk_size: int = 20, 
                 learning_rate: float = 1e-5, 
                 dropout: float = 0.1, 
                 hidden_dim: int = 512, 
                 dim_feedforward: int = 3200, 
                 action_dim: int = 10,
                 camera_names: list = ['ap', 'lateral', 'ap_cropped', 'lateral_cropped'],
                 lr_backbone: float = 1e-5,
                 backbone: str = 'resnet18',
                 enc_layers: int = 4,
                 dec_layers: int = 7,
                 nheads: int = 8,
                 train_full_episode: bool = False,
                 num_epochs: int = 100,
                 ):
        super().__init__()
        # Save hyperparameters for reference (optional)
        self.save_hyperparameters()  
        # Configuration dictionary for ACT model (as args_override in ACT code)
        self.train_full_episode = train_full_episode

        args_override = {
            "lr": learning_rate,
            'num_queries': chunk_size,
            'hidden_dim': hidden_dim,
            'dim_feedforward': dim_feedforward,

            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'enc_layers': enc_layers,
            'dec_layers': dec_layers,
            'nheads': nheads,
            'camera_names': camera_names,
            'action_dim': action_dim,
            'input_channels': 3,

            "kl_weight": kl_weight,
            "chunk_size": chunk_size,
            "dropout": dropout,

            'chkpt_dir': results_dir,
            'policy_class': 'ACT',
            'task_name': TASK_NAME,
            'seed': 0,
            'num_epochs': num_epochs,
            
        }

        # Build the ACT model and optimizer using ACT's own builder function
        # (In the ACT code, ACTPolicy does this via build_ACT_model_and_optimizer)
        # Here we call ACTPolicy to get model and optimizer initialized
        self.policy = ACTPolicy(args_override)  # existing ACT model + optimizer
        self.kl_weight = kl_weight  # store KL weight (also stored inside policy)

    def forward(self, qpos, image):
        """Forward pass for inference (no ground truth actions provided)."""
        # ACTPolicy __call__ returns predicted actions when actions=None (inference mode)
        return self.policy(qpos, image)  # returns a_hat (predicted action chunk)

    def training_step(self, batch, batch_idx):
        """Perform a training step and return loss."""
        qpos, image, actions, is_pad = batch  # unpack batch (assuming this structure)
        # Compute losses using ACTPolicy (returns dict with 'loss', 'l1', 'kl')
        loss_dict = self.policy(qpos, image, actions=actions, is_pad=is_pad)
        loss = loss_dict['loss']
        # Log training metrics
        self.log('train_loss', loss_dict['loss'], prog_bar=True)
        self.log('train_l1', loss_dict['l1'], prog_bar=False)
        self.log('train_kl', loss_dict['kl'], prog_bar=False)
        return loss  # Lightning will use this for optimizer step

    def validation_step(self, batch, batch_idx):
        """Compute validation loss for a batch."""
        qpos_list, image_list, actions_list, is_pad_list = batch
        
        timestep_dicts = []
        for qpos, image, actions, is_pad in zip(qpos_list, image_list, actions_list, is_pad_list):
            timestep_dicts.append(self.policy(qpos, image, actions=actions, is_pad=is_pad))

        loss_dict = compute_dict_mean(timestep_dicts)
        # Log validation metrics (monitor 'val_loss' for early stopping/pruning)
        self.log('val_loss', loss_dict['loss'], prog_bar=True)
        self.log('val_l1', loss_dict['l1'], prog_bar=False)
        self.log('val_kl', loss_dict['kl'], prog_bar=False)
        return loss_dict['loss']

    def configure_optimizers(self):
        """Return the optimizer (and schedulers if any) for training."""
        # The ACTPolicy already created an optimizer in its init
        return self.policy.optimizer

def objective(trial: optuna.Trial, task_name, results_dir) -> float:
    """Optuna objective function: returns the validation loss for a given set of hyperparams."""
    # Suggest hyperparameters from the defined search space
    kl_weight = trial.suggest_categorical('kl_weight', [1, 5, 10, 20, 30])
    # batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    chunk_size = trial.suggest_categorical('chunk_size', [10, 20, 30, 40, 60])
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
    dropout = trial.suggest_categorical('dropout', [0.0, 0.1, 0.2, 0.3, 0.4])
    # train_full_episode = trial.suggest_categorical('train_full_episode', [True, False])

    max_epochs = 15  # set a reasonable upper bound for epochs

    task_config = SIM_TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    train_dir = task_config['train_dir']
    val_dir = task_config['val_dir']
    test_dir = task_config['test_dir']
    num_episodes = task_config['num_episodes']
    episodes_start = task_config['episode_start']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    
    # Create the Lightning model with these hyperparameters
    model = ACTLightningModule(kl_weight=kl_weight, chunk_size=chunk_size, 
                               learning_rate=learning_rate, dropout=dropout, camera_names=camera_names, num_epochs=max_epochs)
    
    # Set up data loaders for this trial
    train_loader, val_loader, _, _ = load_data(dataset_dir, num_episodes, 
                                                           episodes_start, camera_names, 
                                                           batch_size, batch_size, 
                                                           train_dir=train_dir, val_dir=val_dir, 
                                                           episode_len=episode_len, chunk_size=chunk_size
                                                           )
    
    # Set chunk_size in dataset for iterating the full episode
    val_loader.dataset.chunk_size = chunk_size

    # Callbacks: Early stopping and Optuna pruning
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=False)
    pruning_cb = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    
    # Logger (optional): log each trial to a separate TensorBoard log directory
    tb_logger = pl.loggers.TensorBoardLogger("optuna_logs", name=f"trial_{trial.number}")
    
    # Set up the Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,  # set a reasonable upper bound for epochs
        accelerator="gpu", devices=2,  # use 1 GPU per trial (set >1 for multi-GPU training)
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        callbacks=[early_stop, pruning_cb],
        logger=tb_logger,
        enable_progress_bar=False,   # disable verbose progress to keep output clean
        enable_model_summary=False   # disable model summary printout for speed
    )
    
    # Train the model (Lightning will automatically validate at epoch end)
    trainer.fit(model, train_loader, val_loader)
    
    # Retrieve the final validation loss of the last epoch
    final_val_loss = trainer.callback_metrics["val_loss"].item()
    return final_val_loss

# Set up and run the Optuna study
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)  # wait 5 trials and 10 epochs before pruning
study = optuna.create_study(direction="minimize", pruner=pruner)

# Pick your directory
results_dir = "/data_vertebroplasty/flora/vertebroplasty_training/optuna_results"
os.makedirs(results_dir, exist_ok=True)
db_path = os.path.join(results_dir, "act_hpo.db")
TASK_NAME = "NMDID_v2.4"
N_TRIALS = 4  # number of trials to run (can increase for a more exhaustive search)

# --- before optimize ---
storage_uri = f"sqlite:///{db_path}"
study = optuna.create_study(
    study_name="act_hpo",
    direction="minimize",
    pruner=pruner,
    storage=storage_uri,
    load_if_exists=True,
)

study.optimize(lambda t: objective(t, TASK_NAME, results_dir), n_trials=N_TRIALS)

# Print the best result
best_trial = study.best_trial
print(f"Best Validation Loss: {best_trial.value:.4f}")
print("Best Hyperparameters:", best_trial.params)

# Save tables
df = study.trials_dataframe(attrs=("number","state","value","datetime_start","datetime_complete","params"))
df.to_csv("optuna_trials.csv", index=False)

# Save visuals
from optuna.visualization import plot_optimization_history, plot_param_importances
plot_optimization_history(study).write_html(os.path.join(results_dir, "opt_hist.html"))
plot_param_importances(study).write_html(os.path.join(results_dir, "param_importances.html"))
