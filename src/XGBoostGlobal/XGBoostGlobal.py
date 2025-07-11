import os
import yaml
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Optional
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from scipy.stats import pearsonr
import optuna
import xgboost as xgb
import joblib
import shap
from xgboost.callback import EarlyStopping  # Import EarlyStopping callback
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import trange


def to_python_type(obj):
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, (np.generic, np.ndarray)):
        return obj.item()
    else:
        return obj
class XGBoostGridMultiStepPipeline:
    def __init__(self, dataset, config_path: str = "config.yaml"):
        print("Initializing XGBoostGridMultiStepPipeline...")
        self.dataset = dataset
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        self.X = None
        self.Y = None
        self.X_train = self.X_val = self.X_test = None
        self.Y_train = self.Y_val = self.Y_test = None
        self.output_dir = os.path.join("run_outputs", self.config['project_name'])
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_config(self, path: str):
        print(f"Loading configuration from {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def sample_random_patch(self, input_len, forecast_len, spatial_size=(32, 32)):
        data = self.dataset.data
        target = self.dataset.target

        T, F, H, W = data.shape
        max_time = T - input_len - forecast_len
        lat_max = H - spatial_size[0]
        lon_max = W - spatial_size[1]

        t = np.random.randint(0, max_time)
        lat = np.random.randint(0, lat_max)
        lon = np.random.randint(0, lon_max)

        x_patch = data.isel(
            time=slice(t, t + input_len),
            latitude=slice(lat, lat + spatial_size[0]),
            longitude=slice(lon, lon + spatial_size[1])
        ).load()

        y_patch = target.isel(
            time=slice(t + input_len, t + input_len + forecast_len),
            latitude=slice(lat, lat + spatial_size[0]),
            longitude=slice(lon, lon + spatial_size[1])
        ).load()

        x_np = x_patch.transpose("variable", "time", "latitude", "longitude").values
        y_np = y_patch.values

        x_flat = x_np.reshape(x_np.shape[0] * x_np.shape[1], -1).T
        y_flat = y_np.reshape(y_np.shape[0], -1).T

        valid = ~np.isnan(y_flat).any(axis=1)
        return x_flat[valid], y_flat[valid]

    # REPLACE convert_to_tabular WITH PATCH SAMPLING VARIANT
    def convert_to_tabular(self):
        print("Generating patch-based training samples...")
        cache_x = os.path.join(self.output_dir, "X_patch.npy")
        cache_y = os.path.join(self.output_dir, "Y_patch.npy")
        if os.path.exists(cache_x) and os.path.exists(cache_y):
            print("Loading cached patches...")
            self.X = pd.DataFrame(np.load(cache_x))
            self.Y = pd.DataFrame(np.load(cache_y))
            return

        input_len = self.config['input_len']
        forecast_len = self.config['forecast_len']
        num_patches = self.config.get('num_patches', 2)
        spatial_size = tuple(self.config.get('patch_size', [32, 32]))

        X_list, Y_list = [], []
        for _ in trange(num_patches, desc="Sampling patches"):
            x, y = self.sample_random_patch(input_len, forecast_len, spatial_size)
            X_list.append(x)
            Y_list.append(y)

        X = np.vstack(X_list)
        Y = np.vstack(Y_list)

        np.save(cache_x, X)
        np.save(cache_y, Y)

        self.X = pd.DataFrame(X)
        self.Y = pd.DataFrame(Y, columns=[f"{self.config['target_variable']}_t+{i+1}" for i in range(forecast_len)])



    def split_data(self):
        print("Splitting data into train, validation, and test sets...")
        test_size = self.config.get('test_size', 0.2)
        val_size = self.config.get('val_size', 0.1)
        total_len = len(self.X)
        test_len = int(total_len * test_size)
        val_len = int((total_len - test_len) * val_size)
        train_len = total_len - test_len - val_len
        self.X_train = self.X.iloc[:train_len]
        self.Y_train = self.Y.iloc[:train_len]
        self.X_val = self.X.iloc[train_len:train_len + val_len]
        self.Y_val = self.Y.iloc[train_len:train_len + val_len]
        self.X_test = self.X.iloc[train_len + val_len:]
        self.Y_test = self.Y.iloc[train_len + val_len:]

    def scale_data(self):
        print("Scaling data using precomputed global MinMaxScaler...")

        # Load pre-fitted scalers (saved from global Dask scan)
        self.scaler_x = joblib.load(self.config['scaler_x_path'])  # e.g., "scaler_x.joblib"
        self.scaler_y = joblib.load(self.config['scaler_y_path'])  # e.g., "scaler_y.joblib"

        print("[Scaling] Applying global MinMaxScaler...")
        self.X_train = pd.DataFrame(self.scaler_x.transform(self.X_train))
        self.Y_train = pd.DataFrame(self.scaler_y.transform(self.Y_train))
        self.X_val = pd.DataFrame(self.scaler_x.transform(self.X_val))
        self.Y_val = pd.DataFrame(self.scaler_y.transform(self.Y_val))
        self.X_test = pd.DataFrame(self.scaler_x.transform(self.X_test))
        self.Y_test = pd.DataFrame(self.scaler_y.transform(self.Y_test))

        print("[Scaling] X scaler range:", self.scaler_x.data_min_, self.scaler_x.data_max_)
        print("[Scaling] Y scaler range:", self.scaler_y.data_min_, self.scaler_y.data_max_)


    def inverse_transform_predictions(self, preds):
        return self.scaler_y.inverse_transform(preds)

    def train(self):
        model_params = self.config.get('model_params', {})
        base_model = xgb.XGBRegressor(objective='reg:squarederror', **model_params)
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(self.X_train, self.Y_train)
        joblib.dump(self.model, os.path.join(self.output_dir, "xgb_model.joblib"))
    def predict_and_save(self, num_samples: int = 1000):
        preds_scaled = self.model.predict(self.X_test.iloc[:num_samples])
        preds = self.inverse_transform_predictions(preds_scaled)
        true = self.inverse_transform_predictions(self.Y_test.iloc[:num_samples].values)
        df = pd.DataFrame(np.hstack([true, preds]), columns=
                        [f"true_t+{i+1}" for i in range(preds.shape[1])] +
                        [f"pred_t+{i+1}" for i in range(preds.shape[1])])
    ### PATCHED: tune_hyperparameters with safe Optuna objective (no multi-output eval_set)

    def tune_hyperparameters(self):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse'
            }

            model = MultiOutputRegressor(xgb.XGBRegressor(**params))
            model.fit(self.X_train, self.Y_train)  # No early stopping here!
            Y_pred = model.predict(self.X_val)
            rmse = np.mean([
                root_mean_squared_error(self.Y_val.iloc[:, i], Y_pred[:, i])
                for i in range(self.Y_val.shape[1])
            ])
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.config.get("n_trials", 20), show_progress_bar=True)
        best_params = study.best_trial.params

        # Final retraining with early stopping (per output)
        history = {'train': [], 'val': []}
        trained_estimators = []
        print("Retraining best model with early stopping and logging:")

        for i in tqdm(range(self.Y_train.shape[1]), desc="Training output models"):
            evals_result = {}
            estimator = xgb.XGBRegressor(
                **best_params,
                eval_metric='rmse',
                early_stopping_rounds=10,
            )
            estimator.fit(
                self.X_train, self.Y_train.iloc[:, i],
                eval_set=[(self.X_train, self.Y_train.iloc[:, i]), (self.X_val, self.Y_val.iloc[:, i])],
                verbose=False,
            )
            evals_result = estimator.evals_result()
            history['train'].append(evals_result['validation_0']['rmse'])
            history['val'].append(evals_result['validation_1']['rmse'])
            trained_estimators.append(estimator)
        
        self.model = MultiOutputRegressor(estimator=None)
        self.model.estimators_ = trained_estimators

        with open(os.path.join(self.output_dir, "train_val_loss.json"), 'w') as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(self.output_dir, "best_hyperparams.json"), 'w') as f:
            json.dump(best_params, f, indent=2)
        joblib.dump(self.model, os.path.join(self.output_dir, "xgb_model_best.joblib"))


    def evaluate(self, split='test'):
        X_eval = getattr(self, f"X_{split}")
        Y_eval = getattr(self, f"Y_{split}")
        Y_pred_scaled = self.model.predict(X_eval)
        Y_pred = self.inverse_transform_predictions(Y_pred_scaled)
        Y_true = self.inverse_transform_predictions(Y_eval.values)
        metrics = {}
        for i in range(self.Y.shape[1]):
            y_true = Y_true[:, i]
            y_hat = Y_pred[:, i]
            metrics[f"step_{i+1}_rmse"] = root_mean_squared_error(y_true, y_hat)
            metrics[f"step_{i+1}_mae"] = mean_absolute_error(y_true, y_hat)
            metrics[f"step_{i+1}_r2"] = r2_score(y_true, y_hat)
            metrics[f"step_{i+1}_nse"] = 1 - np.sum((y_true - y_hat)**2) / np.sum((y_true - np.mean(y_true))**2)
            hits = (y_hat >= self.config.get("rain_threshold", 0.1))
            actual = (y_true >= self.config.get("rain_threshold", 0.1))
            tp = np.sum(hits & actual)
            fp = np.sum(hits & ~actual)
            fn = np.sum(~hits & actual)
            csi = tp / (tp + fp + fn + 1e-6)
            metrics[f"step_{i+1}_csi"] = csi
        metrics_path = os.path.join(self.output_dir, f"{split}_metrics.json")
        metrics = to_python_type(metrics)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        return metrics

    def plot_train_val_loss(self):
        path = os.path.join(self.output_dir, "train_val_loss.json")
        if not os.path.exists(path):
            print("Loss file not found.")
            return
        with open(path, 'r') as f:
            loss = json.load(f)
        num_outputs = len(loss['train'])
        for i in range(num_outputs):
            plt.figure()
            plt.plot(loss['train'][i], label='Train')
            plt.plot(loss['val'][i], label='Val')
            plt.title(f"Output t+{i+1} RMSE")
            plt.xlabel("Epoch")
            plt.ylabel("RMSE")
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, f"loss_output_t+{i+1}.png"))
            plt.close()
    
    def explain(self, num_samples: int = 1000):
        explainer = shap.Explainer(self.model.estimators_[0])
        shap_values = explainer(self.X_val.iloc[:num_samples])
        shap.summary_plot(shap_values, self.X_val.iloc[:num_samples], show=False)
        plt.savefig(os.path.join(self.output_dir, "shap_summary.png"))

    def load_best_model(self):
        model_path = os.path.join(self.output_dir, "xgb_model_best.joblib")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            raise FileNotFoundError("Best model not found.")
    # ...existing code...


    def run_all(self):
        self.convert_to_tabular()
        self.split_data()
        self.scale_data()
        if self.config.get("enable_tuning", True):
            self.tune_hyperparameters()
        else:
            self.train()
        self.plot_train_val_loss()
        train_metrics = self.evaluate("train")
        print("Training metrics:", train_metrics)
        val_metrics = self.evaluate("val")
        test_metrics = self.evaluate("test")
        print("Validation:", val_metrics)
        print("Test:", test_metrics)
        self.predict_and_save()
        self.explain()
