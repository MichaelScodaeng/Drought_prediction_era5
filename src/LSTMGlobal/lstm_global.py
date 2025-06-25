# Enhanced lstm_pytorch_pipeline_global.py with memory-efficient improvements for large datasets

import os
import torch
import json
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import pickle
import yaml
import psutil
from tqdm import tqdm

from src.GriddedClimateDataset import GriddedClimateDataset

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_scaler(scaler_path):
    """Load a saved scaler"""
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)

def save_scaler(scaler, scaler_path):
    """Save a scaler"""
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

class MemoryEfficientScaling:
    """Memory-efficient scaling for large climate data"""
    
    @staticmethod
    def apply_scaling_chunked(data, scaler, variables, chunk_size=1000):
        """Apply scaling in chunks to avoid memory issues"""
        if data.ndim == 4:  # [C, T, H, W]
            C, T, H, W = data.shape
            scaled_data = np.zeros_like(data)
            
            for c, var in enumerate(variables):
                var_data = data[c]  # [T, H, W]
                
                # Process in spatial chunks
                total_pixels = H * W
                for start_idx in range(0, total_pixels, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_pixels)
                    
                    # Extract chunk
                    flat_data = var_data.reshape(T, -1)  # [T, H*W]
                    chunk = flat_data[:, start_idx:end_idx]  # [T, chunk_size]
                    
                    # Scale chunk
                    chunk_reshaped = chunk.reshape(-1, 1)
                    if hasattr(scaler, 'scalers_') and var in scaler.scalers_:
                        scaled_chunk = scaler.scalers_[var].transform(chunk_reshaped)
                    else:
                        scaled_chunk = scaler.transform(chunk_reshaped)
                    
                    # Put back
                    chunk_scaled = scaled_chunk.reshape(T, -1)
                    flat_scaled = scaled_data[c].reshape(T, -1)
                    flat_scaled[:, start_idx:end_idx] = chunk_scaled
                    scaled_data[c] = flat_scaled.reshape(T, H, W)
            
            return scaled_data
        else:
            return MemoryEfficientScaling._apply_scaling_simple(data, scaler)
    
    @staticmethod
    def _apply_scaling_simple(data, scaler):
        """Fallback for simpler cases"""
        if data.ndim == 3:  # [T, H, W]
            original_shape = data.shape
            data_flat = data.reshape(-1, 1)
            scaled = scaler.transform(data_flat)
            return scaled.reshape(original_shape)
        return data

class MemoryEfficientMetrics:
    """Compute metrics without loading all predictions into memory"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sum_squared_error = 0.0
        self.sum_absolute_error = 0.0
        self.sum_y_true = 0.0
        self.sum_y_pred = 0.0
        self.sum_y_true_squared = 0.0
        self.sum_y_pred_squared = 0.0
        self.sum_y_true_y_pred = 0.0
        self.n_samples = 0
    
    def update(self, y_true, y_pred):
        """Update metrics with a batch"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        batch_size = len(y_true_flat)
        
        self.sum_squared_error += np.sum((y_true_flat - y_pred_flat) ** 2)
        self.sum_absolute_error += np.sum(np.abs(y_true_flat - y_pred_flat))
        self.sum_y_true += np.sum(y_true_flat)
        self.sum_y_pred += np.sum(y_pred_flat)
        self.sum_y_true_squared += np.sum(y_true_flat ** 2)
        self.sum_y_pred_squared += np.sum(y_pred_flat ** 2)
        self.sum_y_true_y_pred += np.sum(y_true_flat * y_pred_flat)
        self.n_samples += batch_size
    
    def compute(self):
        """Compute final metrics"""
        if self.n_samples == 0:
            return {'rmse': 0, 'mae': 0, 'r2': 0}
        
        mse = self.sum_squared_error / self.n_samples
        rmse = np.sqrt(mse)
        mae = self.sum_absolute_error / self.n_samples
        
        # R² calculation
        y_true_mean = self.sum_y_true / self.n_samples
        ss_tot = self.sum_y_true_squared - 2 * y_true_mean * self.sum_y_true + self.n_samples * y_true_mean ** 2
        ss_res = self.sum_squared_error
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }

class MemoryMonitor:
    """Monitor GPU and CPU memory usage"""
    
    @staticmethod
    def print_memory_usage(stage=""):
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_cached = torch.cuda.memory_reserved() / 1024**3   # GB
            print(f"[{stage}] GPU Memory - Allocated: {gpu_memory:.2f}GB, Cached: {gpu_cached:.2f}GB")
        
        cpu_memory = psutil.virtual_memory().used / 1024**3  # GB
        print(f"[{stage}] CPU Memory Used: {cpu_memory:.2f}GB")
    
    @staticmethod
    def clear_cache():
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class LargeScaledGriddedClimateDataset:
    """Memory-efficient version of scaled dataset"""
    
    def __init__(self, base_dataset, scaler=None, variables=None, target_variable=None,
                 scaling_chunk_size=1000):
        self.base_dataset = base_dataset
        self.scaler = scaler
        self.variables = variables
        self.target_variable = target_variable
        self.scaling_chunk_size = scaling_chunk_size
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        
        if self.scaler is not None:
            # Use chunked scaling for memory efficiency
            x = MemoryEfficientScaling.apply_scaling_chunked(
                x, self.scaler, self.variables, self.scaling_chunk_size
            )
            
            if y.ndim == 3:  # [forecast_len, H, W]
                y = MemoryEfficientScaling.apply_scaling_chunked(
                    y.reshape(1, *y.shape), self.scaler, [self.target_variable], 
                    self.scaling_chunk_size
                )[0]
        
        return x, y

def inverse_transform_predictions(predictions, scaler, target_variable):
    """Inverse transform predictions for gridded data"""
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    
    if hasattr(scaler, 'scalers_') and target_variable in scaler.scalers_:
        return scaler.scalers_[target_variable].inverse_transform(predictions)
    else:
        return scaler.inverse_transform(predictions)

class LSTMRegressor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers,
                                   batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class LSTMLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        if x.ndim == 5:  # [B, C, T, H, W]
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B, T, C * H * W)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_flat = y.view(y.size(0), -1)
        loss = self.criterion(y_hat, y_flat)
        self.log("train_loss", loss)
        
        # Clear cache periodically to manage memory
        if batch_idx % 50 == 0:
            MemoryMonitor.clear_cache()
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_flat = y.view(y.size(0), -1)
        loss = self.criterion(y_hat, y_flat)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_flat = y.view(y.size(0), -1)
        loss = self.criterion(y_hat, y_flat)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class LSTMPyTorchGlobalPipeline:
    def __init__(self, config_path="config.yaml"):
        self.cfg = load_config(config_path)
        self.scaler = None
        self.lightning_model = None

    def compute_metrics_efficient(self, loader, name, scaler=None, target_variable=None):
        """Memory-efficient metrics computation with progress bar"""
        self.lightning_model.eval()
        metrics_calculator = MemoryEfficientMetrics()
        
        print(f"\nComputing {name} metrics...")
        MemoryMonitor.print_memory_usage(f"{name} Start")
        
        # Create progress bar
        progress_bar = tqdm(
            loader, 
            desc=f"Evaluating {name}", 
            unit="batch",
            leave=True,
            dynamic_ncols=True
        )
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(progress_bar):
                pred = self.lightning_model(x)
                
                # Move to CPU and convert to numpy
                y_np = y.cpu().numpy()
                pred_np = pred.cpu().numpy()
                
                # Apply inverse scaling if available
                if scaler:
                    try:
                        y_np = self._inverse_transform_batch(y_np, scaler, target_variable)
                        pred_np = self._inverse_transform_batch(pred_np, scaler, target_variable)
                    except Exception as e:
                        tqdm.write(f"Warning: Scaling failed for batch {batch_idx}: {e}")
                
                # Update metrics incrementally
                metrics_calculator.update(y_np, pred_np)
                
                # Update progress bar with current metrics (every 10 batches)
                if batch_idx % 10 == 0 and batch_idx > 0:
                    current_metrics = metrics_calculator.compute()
                    progress_bar.set_postfix({
                        'RMSE': f"{current_metrics['rmse']:.4f}",
                        'MAE': f"{current_metrics['mae']:.4f}",
                        'R²': f"{current_metrics['r2']:.4f}"
                    })
                
                # Clear memory periodically
                if batch_idx % 100 == 0:
                    MemoryMonitor.clear_cache()
        
        # Close progress bar
        progress_bar.close()
        
        metrics = metrics_calculator.compute()
        print(f"{name} Final - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
        MemoryMonitor.print_memory_usage(f"{name} End")
        return metrics
    
    def _inverse_transform_batch(self, data, scaler, target_variable):
        """Apply inverse transform to a batch efficiently"""
        original_shape = data.shape
        data_flat = data.reshape(-1, 1)
        
        if hasattr(scaler, 'scalers_') and target_variable in scaler.scalers_:
            transformed = scaler.scalers_[target_variable].inverse_transform(data_flat)
        else:
            transformed = scaler.inverse_transform(data_flat)
        
        return transformed.reshape(original_shape)

    def run_pipeline(self):
        print("\n--- Memory-Efficient LSTM Global Pipeline with WeatherBench2 ---")
        MemoryMonitor.print_memory_usage("Pipeline Start")

        # Dataset loading
        data_cfg = self.cfg["data"]
        lstm_cfg = self.cfg["lstm_params"]

        base_dataset = GriddedClimateDataset(
            file_path=data_cfg["raw_data_path"],
            input_len=lstm_cfg["n_steps_in"],
            forecast_len=lstm_cfg["n_steps_out"],
            variables=data_cfg["predictor_columns"],
            target_variable=self.cfg["project_setup"]["target_variable"],
            lat_bounds=tuple(data_cfg.get("lat_bounds", (90, -90))),
            lon_bounds=tuple(data_cfg.get("lon_bounds", (0, 360))),
            time_slice=slice(*data_cfg["time_range"])
        )

        # Load scaler if specified
        scaler_path = self.cfg.get("scaling", {}).get("load_scaler_path")
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = load_scaler(scaler_path)
            print(f"Loaded scaler from: {scaler_path}")
        else:
            print("No scaler loaded; assuming data is already normalized.")

        # Wrap dataset with memory-efficient scaling
        scaling_chunk_size = self.cfg.get("scaling", {}).get("chunk_size", 1000)
        dataset = LargeScaledGriddedClimateDataset(
            base_dataset, 
            self.scaler, 
            data_cfg["predictor_columns"],
            self.cfg["project_setup"]["target_variable"],
            scaling_chunk_size
        )

        # Chronological split
        total_len = len(dataset)
        test_size = self.cfg.get("test_size", 0.2)
        val_size = self.cfg.get("val_size", 0.1)
        test_len = int(total_len * test_size)
        val_len = int((total_len - test_len) * val_size)
        train_len = total_len - test_len - val_len
        
        print(f"Dataset splits - Train: {train_len}, Val: {val_len}, Test: {test_len}")
        
        train_set = Subset(dataset, range(0, train_len))
        val_set = Subset(dataset, range(train_len, train_len + val_len))
        test_set = Subset(dataset, range(train_len + val_len, total_len))

        # DataLoaders with memory considerations
        batch_size = lstm_cfg.get("batch_size", 16)
        num_workers = min(4, self.cfg.get("num_workers", 2))  # Limit workers for memory
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, 
                              num_workers=num_workers, pin_memory=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, 
                                num_workers=num_workers, pin_memory=False)

        # Determine model input size
        sample_x, sample_y = dataset[0]
        C, T, H, W = sample_x.shape
        input_size = C * H * W
        
        # Output size should match flattened target shape
        if sample_y.ndim == 3:  # [forecast_len, H, W]
            output_size = sample_y.shape[0] * sample_y.shape[1] * sample_y.shape[2]
        else:
            output_size = sample_y.size
            
        print(f"Model input size: {input_size}, output size: {output_size}")
        MemoryMonitor.print_memory_usage("After Dataset Setup")

        def objective(trial):
            hidden_size = trial.suggest_int("hidden_size", 32, 256)
            n_layers = trial.suggest_int("n_layers", 1, 3)
            dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

            model = LSTMRegressor(input_size, hidden_size, n_layers, output_size, dropout_rate)
            lightning_model = LSTMLightningModule(model, learning_rate)
            print(f"🎮 GPU Available: {torch.cuda.is_available()}")
            print(f"🎮 GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"🎮 Lightning using: {trainer.device}")

            trainer = pl.Trainer(
                max_epochs=lstm_cfg.get("max_epochs", 10),
                callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
                logger=False,
                enable_progress_bar=False,
                accelerator="auto",
                devices=1,
                enable_checkpointing=False
            )
            
            try:
                trainer.fit(lightning_model, train_loader, val_loader)
                MemoryMonitor.clear_cache()  # Clean up after trial
                return trainer.callback_metrics["val_loss"].item()
            except Exception as e:
                print(f"Trial failed: {e}")
                MemoryMonitor.clear_cache()
                return float('inf')

        if lstm_cfg.get("use_optuna", False):
            print("Starting Optuna tuning...")
            MemoryMonitor.print_memory_usage("Before Optuna")
            
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=lstm_cfg.get("n_trials", 10))
            best_params = study.best_params
            print("Best hyperparameters:", best_params)
            
            os.makedirs("outputs", exist_ok=True)
            with open("outputs/best_hyperparams.json", "w") as f:
                json.dump(best_params, f, indent=2)
            print("Best hyperparameters saved to outputs/best_hyperparams.json")
            
            model = LSTMRegressor(
                input_size, 
                best_params["hidden_size"], 
                best_params["n_layers"], 
                output_size, 
                best_params["dropout_rate"]
            )
            self.lightning_model = LSTMLightningModule(model, best_params["learning_rate"])
        else:
            model = LSTMRegressor(
                input_size=input_size,
                hidden_size=lstm_cfg["hidden_size"],
                num_layers=lstm_cfg["n_layers"],
                output_size=output_size,
                dropout_rate=lstm_cfg.get("dropout_rate", 0.2)
            )
            self.lightning_model = LSTMLightningModule(model, learning_rate=lstm_cfg.get("learning_rate", 1e-3))

        # Trainer
        trainer = pl.Trainer(
            max_epochs=lstm_cfg.get("max_epochs", 10),
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5),
                ModelCheckpoint(dirpath="outputs", filename="best_model", monitor="val_loss", save_top_k=1)
            ],
            logger=False,
            enable_progress_bar=True,
            accelerator="auto",
            devices=1
        )

        MemoryMonitor.print_memory_usage("Before Training")
        trainer.fit(self.lightning_model, train_loader, val_loader)
        MemoryMonitor.print_memory_usage("After Training")
        
        trainer.test(self.lightning_model, test_loader)

        # Memory-efficient evaluation
        train_metrics = self.compute_metrics_efficient(
            train_loader, "Train", self.scaler, self.cfg["project_setup"]["target_variable"]
        )
        val_metrics = self.compute_metrics_efficient(
            val_loader, "Validation", self.scaler, self.cfg["project_setup"]["target_variable"]
        )
        test_metrics = self.compute_metrics_efficient(
            test_loader, "Test", self.scaler, self.cfg["project_setup"]["target_variable"]
        )

        # Save model and config
        os.makedirs("outputs", exist_ok=True)
        torch.save(self.lightning_model.state_dict(), "outputs/global_lstm_model.pt")
        
        # Save metrics
        metrics = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
        
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        with open("outputs/global_lstm_config.json", "w") as f:
            json.dump(self.cfg, f, indent=2, default=str)
            
        print("Model, config, and metrics saved to outputs/")
        MemoryMonitor.print_memory_usage("Pipeline End")

        return "Training complete."

# Optional run (uncomment below to run as script)
# if __name__ == "__main__":
#     pipeline = LSTMPyTorchGlobalPipeline("config.yaml")
#     pipeline.run_pipeline()