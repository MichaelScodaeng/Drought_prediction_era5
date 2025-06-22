import os
import yaml
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Optional
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import optuna
import xgboost as xgb
import joblib
import shap
import matplotlib.pyplot as plt

class XGBoostGridMultiStepPipeline:
    def __init__(self, dataset, config_path: str = "config.yaml"):
        self.dataset = dataset
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.model = None
        self.X = None
        self.Y = None
        self.X_train = self.X_val = self.X_test = None
        self.Y_train = self.Y_val = self.Y_test = None

        self.output_dir = os.path.join("run_outputs", self.config['project_name'])
        os.makedirs(self.output_dir, exist_ok=True)
        print("Kuy output directory:", self.output_dir)

    def _load_config(self, path: str):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def convert_to_tabular(self):
        print("Converting dataset to tabular format...")
        input_len = self.config['input_len']
        forecast_len = self.config['forecast_len']
        max_samples = self.config.get('max_samples', None)

        data = self.dataset.data.values  # [T, F, H, W]
        target = self.dataset.target.values  # [T, H, W]
        T, F, H, W = data.shape
        num_windows = T - input_len - forecast_len + 1

        X_list, Y_list = [], []

        for t in tqdm(range(num_windows), desc="Creating tabular dataset"):
            x_seq = data[t:t+input_len]  # [input_len, F, H, W]
            y_seq = target[t+input_len:t+input_len+forecast_len]  # [forecast_len, H, W]

            x_flat = x_seq.transpose(1, 0, 2, 3).reshape(F * input_len, H * W).T
            y_flat = y_seq.reshape(forecast_len, H * W).T

            X_list.append(x_flat)
            Y_list.append(y_flat)

        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)

        if max_samples and X.shape[0] > max_samples:
            idx = np.random.choice(X.shape[0], max_samples, replace=False)
            X, Y = X[idx], Y[idx]

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
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
            }

            model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', early_stopping_rounds=10))
            model.fit(self.X_train, self.Y_train, 
                      eval_set=[(self.X_val, self.Y_val)], 
                      verbose=False)
            Y_pred = model.predict(self.X_val)

            rmse = np.mean([mean_squared_error(self.Y_val.iloc[:, i], Y_pred[:, i], squared=False) for i in range(self.Y_val.shape[1])])
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.config.get("n_trials", 20), show_progress_bar=True)

        best_params = study.best_trial.params
        self.model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', **best_params))
        history = {'train': [], 'val': []}

        print("Retraining best model with early stopping and logging:")
        for i, estimator in enumerate(tqdm(self.model.estimators_, desc="Training output models")):
            evals_result = {}
            estimator.fit(self.X_train, self.Y_train.iloc[:, i], 
                          eval_set=[(self.X_train, self.Y_train.iloc[:, i]), (self.X_val, self.Y_val.iloc[:, i])], 
                          early_stopping_rounds=10, 
                          eval_metric='rmse', 
                          verbose=False,
                          callbacks=[xgb.callback.EvaluationMonitor(evals_result)])
            history['train'].append(evals_result['validation_0']['rmse'])
            history['val'].append(evals_result['validation_1']['rmse'])

        with open(os.path.join(self.output_dir, "train_val_loss.json"), 'w') as f:
            json.dump(history, f, indent=2)

        with open(os.path.join(self.output_dir, "best_hyperparams.json"), 'w') as f:
            json.dump(best_params, f, indent=2)
        joblib.dump(self.model, os.path.join(self.output_dir, "xgb_model_best.joblib"))

    def train(self):
        model_params = self.config.get('model_params', {})
        base_model = xgb.XGBRegressor(objective='reg:squarederror', **model_params)
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(self.X_train, self.Y_train)
        joblib.dump(self.model, os.path.join(self.output_dir, "xgb_model.joblib"))

    def evaluate(self, split='test'):
        X_eval = getattr(self, f"X_{split}")
        Y_eval = getattr(self, f"Y_{split}")
        Y_pred = self.model.predict(X_eval)

        metrics = {}
        for i in range(self.Y.shape[1]):
            y_true = Y_eval.iloc[:, i].values
            y_hat = Y_pred[:, i]
            metrics[f"step_{i+1}_rmse"] = mean_squared_error(y_true, y_hat, squared=False)
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
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        return metrics

    def predict_and_save(self, num_samples: int = 1000):
        preds = self.model.predict(self.X_test.iloc[:num_samples])
        pred_df = pd.DataFrame(preds, columns=[f"pred_t+{i+1}" for i in range(preds.shape[1])])
        pred_df.to_csv(os.path.join(self.output_dir, "sample_predictions.csv"), index=False)

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

    def run_all(self):
        self.convert_to_tabular()
        self.split_data()
        if self.config.get("enable_tuning", True):
            self.tune_hyperparameters()
        else:
            self.train()
        val_metrics = self.evaluate("val")
        test_metrics = self.evaluate("test")
        print("Validation:", val_metrics)
        print("Test:", test_metrics)
        self.predict_and_save()
        self.explain()
