import os
import optuna

# Pick your directory
results_dir = "/data_vertebroplasty/flora/vertebroplasty_training/optuna_results"
db_path = os.path.join(results_dir, "act_hpo.db")

# --- before optimize ---
storage_uri = f"sqlite:///{db_path}"

study = optuna.load_study(
    study_name="act_hpo_t11",
    storage=storage_uri
)


# Save tables
df = study.trials_dataframe(attrs=("number","state","value","datetime_start","datetime_complete","params"))
df.to_csv(os.path.join(results_dir, "optuna_trials.csv"), index=False)

# Save visuals
from optuna.visualization import plot_optimization_history, plot_param_importances
plot_optimization_history(study).write_html(os.path.join(results_dir, "opt_hist.html"))
plot_param_importances(study).write_html(os.path.join(results_dir, "param_importances.html"))
