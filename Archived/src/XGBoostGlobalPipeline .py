import pandas as pd
import numpy as np
import yaml
import os
import joblib
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt # For feature importance
import json
# Assuming these are in src/ or PYTHONPATH is set for the notebook
try:
    from src.data_utils import load_config, load_and_prepare_data, split_data_chronologically
    from src.preprocess_utils import scale_data, save_scaler, load_scaler, inverse_transform_predictions
    from src.feature_utils import engineer_features
    print("Pipeline Class: Successfully imported utility functions.")
except ImportError as e:
    print(f"Pipeline Class Error: Could not import utility functions: {e}")
    print("Ensure your PYTHONPATH is set correctly if running from a notebook, or that src is accessible.")
    # Define dummy functions if import fails, so class can be parsed
    def load_config(path=None): return {}
    def load_and_prepare_data(config=None): return None
    def split_data_chronologically(df=None, config=None): return None, None, None
    def engineer_features(df=None, config=None): return df
    def scale_data(df_train=None, df_val=None, df_test=None, config=None): return None,None,None,None
    def save_scaler(scaler=None, path=None): pass
    def load_scaler(path=None): return None
    def inverse_transform_predictions(df=None, target=None, scaler=None): return None


import pandas as pd
import numpy as np
import yaml
import os
import joblib
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt # For feature importance
import json # For saving metrics

# Assuming these are in src/ or PYTHONPATH is set for the notebook
try:
    from src.data_utils import load_config, load_and_prepare_data, split_data_chronologically
    from src.preprocess_utils import scale_data, save_scaler, load_scaler, inverse_transform_predictions
    from src.feature_utils import engineer_features
    print("Pipeline Class: Successfully imported utility functions.")
except ImportError as e:
    print(f"Pipeline Class Error: Could not import utility functions: {e}")
    print("Ensure your PYTHONPATH is set correctly if running from a notebook, or that src is accessible.")
    # Define dummy functions if import fails, so class can be parsed
    def load_config(path=None): return {}
    def load_and_prepare_data(config=None): return None
    def split_data_chronologically(df=None, config=None): return None, None, None
    def engineer_features(df=None, config=None): return df
    def scale_data(df_train=None, df_val=None, df_test=None, config=None): return None,None,None,None
    def save_scaler(scaler=None, path=None): pass
    def load_scaler(path=None): return None
    def inverse_transform_predictions(df=None, target=None, scaler=None): return None


class XGBoostGlobalPipeline:
    def __init__(self, config_path="config.yaml"):
        self.config_path_abs = os.path.abspath(config_path)
        print(f"Pipeline Class: Attempting to load config from: {self.config_path_abs}")
        self.cfg = load_config(self.config_path_abs)
        
        if not self.cfg or self.cfg.get('data',{}).get('raw_data_path') is None:
            print("Pipeline Class Warning: Configuration might not have loaded correctly. Critical paths might be missing.")

        self.scaler = None
        self.model = None
        self.best_hyperparams = None
        self.train_df_raw, self.val_df_raw, self.test_df_raw = None, None, None
        self.train_df_featured, self.val_df_featured, self.test_df_featured = None, None, None
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = [None]*6
        self.full_df_raw_for_prediction = None # Initialize
        
        self.experiment_name = self.cfg.get('project_setup', {}).get('project_name', 'default_experiment')
        self.project_root_for_paths = os.path.dirname(self.config_path_abs) # Directory of config file

        results_base_cfg = self.cfg.get('paths',{}).get('output_base_dir', 'run_outputs')
        self.run_output_dir = os.path.join(self.project_root_for_paths, results_base_cfg, self.experiment_name)
        
        models_base_dir_cfg = self.cfg.get('paths', {}).get('models_base_dir', 'models_saved') # Changed to 'paths.models_base_dir'
        self.run_models_dir = os.path.join(self.project_root_for_paths, models_base_dir_cfg, self.experiment_name)

        os.makedirs(self.run_output_dir, exist_ok=True)
        os.makedirs(self.run_models_dir, exist_ok=True)
        print(f"Pipeline Class: Artifacts for experiment '{self.experiment_name}' will be saved under '{self.run_output_dir}' and '{self.run_models_dir}'")


    def _get_abs_path_from_config_value(self, relative_path_from_config_value):
        if relative_path_from_config_value is None: return None
        if os.path.isabs(relative_path_from_config_value): return relative_path_from_config_value
        return os.path.abspath(os.path.join(self.project_root_for_paths, relative_path_from_config_value))

    def load_and_split_data(self):
        print("Pipeline: Loading and splitting data...")
        relative_raw_data_path = self.cfg.get('data', {}).get('raw_data_path')
        if not relative_raw_data_path:
            print("Pipeline Error: 'data.raw_data_path' not found in configuration.")
            return
        abs_data_file_path = self._get_abs_path_from_config_value(relative_raw_data_path)
        if not abs_data_file_path or not os.path.exists(abs_data_file_path):
            print(f"Pipeline Error: Data file not found at constructed absolute path: {abs_data_file_path}")
            return

        temp_load_cfg = self.cfg.copy(); temp_load_cfg['data'] = self.cfg['data'].copy() 
        temp_load_cfg['data']['raw_data_path'] = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
        temp_load_cfg['data']['start_date'] = "2021-01-01"
        temp_load_cfg['data']['end_date'] = "2021-03-31"
        temp_load_cfg['data']['lon_bounds'] = [75, 30]
        full_df_raw = load_and_prepare_data(temp_load_cfg) 
        if full_df_raw is None:
            print("Pipeline Error: data_utils.load_and_prepare_data returned None."); return
        self.full_df_raw_for_prediction = full_df_raw.copy() 
        self.train_df_raw, self.val_df_raw, self.test_df_raw = split_data_chronologically(full_df_raw, self.cfg)
        print(f"Pipeline: Data loaded and split. Train shape: {self.train_df_raw.shape if self.train_df_raw is not None else 'None'}")

    def engineer_all_features(self):
        print("Pipeline: Engineering features...")
        if self.train_df_raw is None: raise ValueError("Raw training data not loaded for feature engineering.")
        self.train_df_featured = engineer_features(self.train_df_raw.copy(), self.cfg)
        self.val_df_featured = engineer_features(self.val_df_raw.copy(), self.cfg)
        self.test_df_featured = engineer_features(self.test_df_raw.copy(), self.cfg)
        #print shape all
        print(f"Pipeline: Feature engineering complete. Train shape: {self.train_df_featured.shape if self.train_df_featured is not None else 'None'}, "
                f"Validation shape: {self.val_df_featured.shape if self.val_df_featured is not None else 'None'}, "
                f"Test shape: {self.test_df_featured.shape if self.test_df_featured is not None else 'None'}")

    def preprocess_all_data(self):
        print("Pipeline: Scaling data...")
        if self.train_df_featured is None: raise ValueError("Featured training data not available for scaling.")
        scaled_train, scaled_val, scaled_test, fitted_sclr = scale_data(
            self.train_df_featured, self.val_df_featured, self.test_df_featured, self.cfg)
        if fitted_sclr is None: raise ValueError("Scaler fitting failed.")
        self.scaler = fitted_sclr
        
        target_col = self.cfg['project_setup']['target_variable']
        time_col = self.cfg['data']['time_column']
        cols_to_drop_for_X = [target_col]
        if time_col in scaled_train.columns: cols_to_drop_for_X.append(time_col)
        #drop any columns that are not contains "lag" or "rolling" in their name
        cols_to_drop_for_X += [col for col in scaled_train.columns if 'lag' not in col and 'rolling' not in col and col!="year" and col!="month" and col!="day" and col!="lat" and col!="lon"]
        self.X_train = scaled_train.drop(columns=cols_to_drop_for_X, errors='ignore')
        print("Columns: ", self.X_train.columns.tolist()) # DEBUG PRINT
        self.y_train = scaled_train[target_col]
        self.X_val = scaled_val.drop(columns=cols_to_drop_for_X, errors='ignore')
        self.y_val = scaled_val[target_col]
        self.X_test = scaled_test.drop(columns=cols_to_drop_for_X, errors='ignore')
        self.y_test = scaled_test[target_col]


        scaler_filename = self.cfg.get('scaling',{}).get('scaler_filename', 'robust_scaler.joblib')
        scaler_save_path = os.path.join(self.run_models_dir, scaler_filename) 
        save_scaler(self.scaler, scaler_save_path)
        print(f"Pipeline: Data scaling and X,y preparation complete. Scaler saved to {scaler_save_path}")

    def _objective_for_optuna(self, trial):
        target_col = self.cfg['project_setup']['target_variable']
        param = {
            'objective': self.cfg.get('model_params', {}).get('global_xgboost', {}).get('objective', 'reg:squarederror'),
            'eval_metric': self.cfg.get('model_params', {}).get('global_xgboost', {}).get('eval_metric', 'rmse'),
            'tree_method': 'hist', 'random_state': self.cfg.get('project_setup', {}).get('random_seed', 42),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        }
        model = xgb.XGBRegressor(**param,early_stopping_rounds = 2)
        fit_params_opt = {'verbose': False}
        if xgb.__version__ >= '0.90': 
            fit_params_opt['eval_set'] = [(self.X_val, self.y_val)] 
        model.fit(self.X_train, self.y_train, **fit_params_opt) 
        preds_val_scaled = model.predict(self.X_val) 
        scaled_preds_val_df_opt = pd.DataFrame(preds_val_scaled, columns=[target_col], index=self.X_val.index)
        inversed_predictions_val_opt = inverse_transform_predictions(scaled_preds_val_df_opt, target_col, self.scaler)
        scaled_actuals_val_df_opt = pd.DataFrame(self.y_val.values, columns=[target_col], index=self.y_val.index)
        inversed_actuals_val_opt = inverse_transform_predictions(scaled_actuals_val_df_opt, target_col, self.scaler)
        if inversed_predictions_val_opt is None or inversed_actuals_val_opt is None: return float('inf')
        return mean_squared_error(inversed_actuals_val_opt, inversed_predictions_val_opt)

    def tune_hyperparameters(self, n_trials=50):
        print("Pipeline: Tuning hyperparameters...")
        if self.X_train is None: raise ValueError("Data not preprocessed for hyperparameter tuning.")
        print(f"Pipeline: Starting hyperparameter tuning with {n_trials} trials...")
        print(f"Pipeline: Using {self.X_train.shape} training samples, {self.X_val.shape} validation samples.")
        print("Columns in X_train:", self.X_train.columns.tolist()) # DEBUG PRINT
        print("Columns in X_val:", self.X_val.columns.tolist())
        print("Target variable:", self.cfg['project_setup']['target_variable']) # DEBUG PRINT
        print(" in X_train:", self.X_train.columns.tolist()) # DEBUG PRINT
        study = optuna.create_study(direction='minimize')
        study.optimize(self._objective_for_optuna, n_trials=n_trials)
        self.best_hyperparams = study.best_trial.params
        print(f"Pipeline: Hyperparameter tuning complete. Best RMSE on validation: {study.best_trial.value:.4f}")
        print(f"Best params: {self.best_hyperparams}")

    def train_final_model(self, params=None):
        print("Pipeline: Training final model...")
        if self.X_train is None: raise ValueError("Data not preprocessed for final model training.")
        model_params_to_use = params if params else self.best_hyperparams
        if not model_params_to_use:
            print("Pipeline Warning: No best hyperparameters. Using initial defaults from config.")
            model_params_to_use = self.cfg.get('model_params', {}).get('global_xgboost', {}).copy(); model_params_to_use.pop('tuning', None) 
        final_xgb_model_params = {
            'objective': self.cfg.get('model_params', {}).get('global_xgboost', {}).get('objective', 'reg:squarederror'),
            'eval_metric': self.cfg.get('model_params', {}).get('global_xgboost', {}).get('eval_metric', 'rmse'),
            'tree_method': 'hist', 'random_state': self.cfg.get('project_setup', {}).get('random_seed', 42),
            **model_params_to_use }
        self.model = xgb.XGBRegressor(**final_xgb_model_params)
        print(f"Training final model on X_train (shape: {self.X_train.shape})")
        #remove target from X_train if it exists
        print(self.X_train.columns.tolist()) # DEBUG PRINT
        print("========== DEBUG: X_train Head ==========")
        print(self.X_train.head())
        print("X_train shape:", self.X_train.shape)
        print("X_train columns:", self.X_train.columns.tolist())

        print("\n========== DEBUG: y_train Head ==========")
        print(self.y_train.head())
        print("y_train shape:", self.y_train.shape)
        print("y_train descriptive stats:")
        print(self.y_train.describe())

        # Also check for NaNs
        print("\n========== DEBUG: Check NaNs ==========")
        print("X_train has NaNs:", self.X_train.isna().sum().sum() > 0)
        print("y_train has NaNs:",self.y_train.isna().sum() > 0)
        print("full shape", self.X_train.shape, "y shape:", self.y_train.shape) # DEBUG PRINT

        self.model.fit(self.X_train, self.y_train, verbose=False) 
        self.save_model() 
        print("Pipeline: Final model trained and saved.")

    def save_model(self):
        if self.model is None: print("Pipeline Error: No model to save."); return
        model_filename = self.cfg.get('model_params',{}).get('global_xgboost',{}).get('model_filename', 'xgboost_model.json')
        model_save_path = os.path.join(self.run_models_dir, model_filename)
        try:
            self.model.save_model(model_save_path) 
            print(f"Pipeline: XGBoost model saved to {model_save_path}")
        except Exception as e:
            print(f"Pipeline Error: Could not save XGBoost model to {model_save_path}: {e}")

    def evaluate(self, data_split='test'):
        print(f"Pipeline: Evaluating model on {data_split} set...")
        if self.model is None: print("Pipeline Error: Model not trained."); return None
        if self.scaler is None: print("Pipeline Error: Scaler not available."); return None

        X_eval, y_eval_scaled = None, None
        if data_split == 'test' and self.X_test is not None: X_eval, y_eval_scaled = self.X_test, self.y_test
        elif data_split == 'validation' and self.X_val is not None: X_eval, y_eval_scaled = self.X_val, self.y_val
        elif data_split == 'train' and self.X_train is not None: X_eval, y_eval_scaled = self.X_train, self.y_train
        else: print(f"Pipeline Error: Data for split '{data_split}' unavailable."); return None
        
        scaled_predictions = self.model.predict(X_eval)
        target_col = self.cfg['project_setup']['target_variable']
        scaled_actuals_df = pd.DataFrame(y_eval_scaled.values, columns=[target_col], index=y_eval_scaled.index)
        scaled_preds_df = pd.DataFrame(scaled_predictions, columns=[target_col], index=y_eval_scaled.index)
        inversed_predictions = inverse_transform_predictions(scaled_preds_df, target_col, self.scaler)
        inversed_actuals = inverse_transform_predictions(scaled_actuals_df, target_col, self.scaler)
        
        if inversed_predictions is not None and inversed_actuals is not None:
            from sklearn.metrics import root_mean_squared_error
            rmse = root_mean_squared_error(inversed_actuals, inversed_predictions)
            mae = mean_absolute_error(inversed_actuals, inversed_predictions)
            r2 = r2_score(inversed_actuals, inversed_predictions)
            print(f"{data_split.capitalize()} Set Evaluation (Original Scale): RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            return {'rmse': rmse, 'mae': mae, 'r2': r2}
        else: print(f"Pipeline Error: Could not inverse transform {data_split} predictions/actuals."); return None

    def generate_and_save_feature_importance(self):
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            print("Pipeline Warning: Model not trained or doesn't support feature importance. Skipping plot.")
            return
        if self.X_train is None or self.X_train.empty:
            print("Pipeline Warning: X_train is not available. Cannot map feature importances to names. Skipping plot.")
            return

        feat_imp_filename = self.cfg.get('results',{}).get('feature_importance_filename', 'feature_importance.png')
        plot_save_path = os.path.join(self.run_output_dir, feat_imp_filename)
        try:
            fig, ax = plt.subplots(figsize=(10, max(8, len(self.X_train.columns) * 0.25))) 
            xgb.plot_importance(self.model, ax=ax, max_num_features=20, height=0.8, importance_type='weight') 
            ax.set_title(f"XGBoost Feature Importance ({self.experiment_name})")
            plt.tight_layout()
            plt.savefig(plot_save_path)
            plt.close(fig) 
            print(f"Pipeline: Feature importance plot saved to {plot_save_path}")
        except Exception as e:
            print(f"Pipeline Error: Could not generate/save feature importance plot: {e}")

    def predict_on_full_data(self):
        print("Pipeline: Generating predictions on the full raw dataset...")
        if self.model is None or self.scaler is None:
            print("Pipeline Error: Model or scaler not available. Cannot make full data predictions.")
            return None
        if self.full_df_raw_for_prediction is None: 
            print("Pipeline Error: Original full raw dataframe copy not available for prediction.")
            return None

        print("  Engineering features for full dataset...")
        self.full_df_raw_for_prediction.sort_values(by=self.cfg['data']['time_column'], inplace=True) # Ensure time order
        full_df_featured = engineer_features(self.full_df_raw_for_prediction.copy(), self.cfg)
        self.full_df_raw_for_prediction = full_df_featured.copy() # Update the original copy with featured data
        if full_df_featured.empty:
            print("Pipeline Error: Feature engineering on full dataset resulted in an empty DataFrame.")
            return None
        
        print(f"  Columns in full_df_featured after engineering: {full_df_featured.columns.tolist()}") # DEBUG PRINT

        time_col = self.cfg['data']['time_column']
        target_col_name = self.cfg['project_setup']['target_variable']
        
        scaler_feature_names = list(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else []
        if not scaler_feature_names:
            print("Pipeline Error: Scaler has no feature_names_in_. Was it fitted correctly on named features?")
            return None

        # Create a DataFrame with only the columns the scaler was fitted on, in that order
        df_to_scale_full = pd.DataFrame(index=full_df_featured.index)
        for col in scaler_feature_names:
            if col in full_df_featured:
                df_to_scale_full[col] = full_df_featured[col]
            else:
                # This means a column the scaler expects is missing after feature engineering the full data.
                print(f"Pipeline Warning: Column '{col}' (expected by scaler) not found in feature-engineered full data. Filling with NaN.")
                df_to_scale_full[col] = np.nan # Scaler might handle NaNs (e.g. RobustScaler ignores them) or fail.

        print("  Scaling features for full dataset...")
        scaled_values_for_subset = self.scaler.transform(df_to_scale_full[scaler_feature_names])
        scaled_subset_df = pd.DataFrame(scaled_values_for_subset, columns=scaler_feature_names, index=df_to_scale_full.index)

        # Now, construct X_full_for_prediction using self.X_train.columns as the template
        # It should contain:
        # 1. Scaled versions of columns that were in scaler_feature_names
        # 2. Original (unscaled) versions of other columns that are in X_train.columns (e.g. lat, lon, month, year)
        
        X_full_for_prediction = pd.DataFrame(index=full_df_featured.index)
        print(f"  Model expects columns: {self.X_train.columns.tolist()}") # DEBUG PRINT

        for col in self.X_train.columns:
            if col in scaled_subset_df.columns: # If it was a column that got scaled
                X_full_for_prediction[col] = scaled_subset_df[col]
            elif col in full_df_featured.columns: # If it's an unscaled feature (like lat, lon, month, year)
                X_full_for_prediction[col] = full_df_featured[col]
            else:
                print(f"Pipeline CRITICAL Warning: Feature '{col}' expected by model not found in any processed full data source. Filling with 0.")
                X_full_for_prediction[col] = 0 # Fallback: not ideal

        print(f"  Shape of X_full_for_prediction before predict: {X_full_for_prediction.shape}")
        print(f"  Columns in X_full_for_prediction before predict: {X_full_for_prediction.columns.tolist()}") # DEBUG PRINT

        print("  Making predictions...")
        scaled_predictions_full = self.model.predict(X_full_for_prediction)

        print("  Inverse transforming predictions...")
        scaled_preds_full_df = pd.DataFrame(scaled_predictions_full, columns=[target_col_name], index=full_df_featured.index)
        inversed_predictions_full = inverse_transform_predictions(scaled_preds_full_df, target_col_name, self.scaler)

        if inversed_predictions_full is not None:
            # Start with original full_df_raw_for_prediction to keep original columns and correct length before feature engineering NaNs were dropped
            # Then merge predictions based on index.
            # The index of inversed_predictions_full matches full_df_featured (after NaN drop).
            # So, we need to add predictions to full_df_featured first, then decide what to merge back to the true original.
            
            output_df_with_predictions = full_df_featured.copy()
            output_df_with_predictions[f'{target_col_name}_predicted'] = inversed_predictions_full.values # .values to align if index is slightly off

            # What to save? We want original time, lat, lon, original spei (if available), and predicted spei.
            # The full_df_raw_for_prediction has the original length and all original data.
            # We can merge our predictions (which are on the reduced length full_df_featured index) back to full_df_raw_for_prediction.
            
            final_output_df = self.full_df_raw_for_prediction.copy()
            # Add the prediction where indexes match. Non-matching will be NaN.
            final_output_df = final_output_df.merge(
                output_df_with_predictions[[f'{target_col_name}_predicted']], # Only the prediction column
                left_index=True,
                right_index=True,
                how='left' # Keep all original rows, add predictions where available
            )


            cols_to_save = [time_col, 'lat', 'lon']
            if target_col_name in final_output_df.columns: 
                cols_to_save.append(target_col_name)
            cols_to_save.append(f'{target_col_name}_predicted')
            for orig_pred_col in ['pre','tmp']: # Example other original columns
                if orig_pred_col in final_output_df.columns:
                     cols_to_save.append(orig_pred_col)
            
            final_output_df_subset = final_output_df[[col for col in cols_to_save if col in final_output_df.columns]]

            pred_filename = self.cfg.get('results',{}).get('predictions_filename', 'full_data_predictions.csv')
            save_path = os.path.join(self.run_output_dir, pred_filename)
            try:
                final_output_df_subset.to_csv(save_path, index=False)
                print(f"Pipeline: Full data predictions saved to {save_path}")
                return final_output_df_subset
            except Exception as e:
                print(f"Pipeline Error: Could not save full data predictions: {e}")
        else:
            print("Pipeline Error: Failed to inverse transform full data predictions.")
        return None

    def save_run_config(self):
        config_filename = self.cfg.get('results',{}).get('config_filename', 'config_used.yaml')
        save_path = os.path.join(self.run_output_dir, config_filename)
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.cfg, f, default_flow_style=False, sort_keys=False)
            print(f"Pipeline: Configuration used for this run saved to {save_path}")
        except Exception as e:
            print(f"Pipeline Error: Could not save run configuration: {e}")

    def run_full_pipeline(self, tune=True, n_trials_tuning=200):
        print(f"\n--- Starting Pipeline Run: Experiment '{self.experiment_name}' ---")
        self.load_and_split_data()
        if self.train_df_raw is None: print("Pipeline Halted: Failed at data loading/splitting."); return "Failed: Data Load/Split"
        
        self.engineer_all_features()
        if self.train_df_featured is None or self.train_df_featured.empty : print("Pipeline Halted: Failed at feature engineering."); return "Failed: Feature Engineering"
        
        self.preprocess_all_data()
        if self.X_train is None: print("Pipeline Halted: Failed at data preprocessing/scaling."); return "Failed: Preprocessing"

        if tune:
            self.tune_hyperparameters(n_trials=n_trials_tuning)
        
        self.train_final_model() 
        if self.model is None: print("Pipeline Halted: Failed at final model training."); return "Failed: Model Training"

        self.generate_and_save_feature_importance()
        self.save_run_config() 

        all_metrics = {}
        print("\n--- Final Model Evaluation ---")
        for split_name in ['train', 'validation', 'test']:
            metrics = self.evaluate(data_split=split_name)
            if metrics: all_metrics[split_name] = metrics
        
        metrics_filename = self.cfg.get('results',{}).get('metrics_filename', 'evaluation_metrics.json')
        metrics_save_path = os.path.join(self.run_output_dir, metrics_filename)
        try:
            with open(metrics_save_path, 'w') as f:
                json.dump(all_metrics, f, indent=4) 
            print(f"Pipeline: Evaluation metrics saved to {metrics_save_path}")
        except Exception as e:
            print(f"Pipeline Error: Could not save metrics: {e}")

        self.predict_on_full_data() 

        print(f"--- Pipeline Run Finished: Experiment '{self.experiment_name}' ---")
        return all_metrics




