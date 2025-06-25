import xarray as xr
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
from cftime import num2date

class GriddedClimateDataset(Dataset):
    def __init__(self, file_path, input_len=12, forecast_len=1, variables=None,
                 target_variable=None, lat_bounds=(21.0, 5.0), lon_bounds=(97, 106),
                 time_slice=slice("2020-01", "2023-01")):
        
        if file_path.endswith(".nc"):
            self.ds = xr.open_dataset(file_path)
        else:
            self.ds = xr.open_zarr(file_path, consolidated=True, decode_times=False,
                                   storage_options={"token": "anon", "asynchronous": True})
            from cftime import num2date
            units = self.ds.time.attrs.get("units", "hours since 1900-01-01 00:00:0.0")
            cal = self.ds.time.attrs.get("calendar", "standard")
            self.ds["time"] = ("time", num2date(self.ds.time.values, units, calendar=cal))

        # Subset region and time
        self.ds = self.ds.sel(time=time_slice, latitude=slice(*lat_bounds))
        self.ds = self.ds.where((self.ds.longitude >= lon_bounds[0]) | 
                                (self.ds.longitude <= lon_bounds[1]), drop=True)

        # Process each variable to ensure 3D (time, lat, lon)
        processed_vars = []
        for var in variables:
            da = self.ds[var]
            if "level" in da.dims:
                da = da.mean(dim="level")  # reduce level
            if set(da.dims) == {"latitude", "longitude"}:  # static 2D
                da = da.expand_dims(time=self.ds.time)
            processed_vars.append(da)

        # Stack into [time, variable, lat, lon]
        stacked = xr.concat(processed_vars, dim="variable")
        stacked = stacked.transpose("time", "variable", "latitude", "longitude")

        self.data = stacked
        self.target = self.ds[target_variable].transpose("time", "latitude", "longitude")
        self.time_coords = self.ds.time.values
        self.input_len = input_len
        self.forecast_len = forecast_len
        self.variables = variables
        self.target_variable = target_variable
        self.length = len(self.data.time) - input_len - forecast_len + 1
        print(f"Dataset initialized with {self.length} samples.")



# --- Example Training Loop for ConvLSTM / MESA-NET Compatible ---
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)  # Expected output shape: [B, forecast_len, H, W]
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

'''# Usage
INPUT_LEN = 12  # Number of input time steps
FORECAST_LEN = 6  # Number of forecast time steps
PATH = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
VARIABLES = [
    'total_precipitation_6hr',
    '2m_temperature', '2m_dewpoint_temperature', 'surface_pressure',
    'mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind',
    '10m_wind_speed', 'u_component_of_wind', 'v_component_of_wind',
    'total_column_water_vapour', 'integrated_vapor_transport', 'boundary_layer_height',
    'specific_humidity', 'total_cloud_cover',
    'mean_surface_net_short_wave_radiation_flux',
    'mean_surface_latent_heat_flux', 'mean_surface_sensible_heat_flux',
    'snow_depth', 'sea_surface_temperature', 'volumetric_soil_water_layer_1',
    'mean_vertically_integrated_moisture_divergence', 'eddy_kinetic_energy',
    'land_sea_mask'
]
TARGET_VARIABLE = 'total_precipitation_6hr'
dataset = GriddedClimateDataset(PATH, input_len=INPUT_LEN, forecast_len=FORECAST_LEN, 
                                  variables=VARIABLES, target_variable=TARGET_VARIABLE)'''