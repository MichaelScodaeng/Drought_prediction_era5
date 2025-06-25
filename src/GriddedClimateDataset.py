import xarray as xr
import numpy as np
from torch.utils.data import Dataset

class GriddedClimateDataset(Dataset):
    def __init__(self, file_path, input_len=12, forecast_len=1, variables=None,
                 target_variable=None, lat_bounds=(21.0, 5.0), lon_bounds=(97, 106),
                 time_slice=slice("2022-01", "2022-02")):
        self.input_len = input_len
        self.forecast_len = forecast_len
        self.variables = variables
        self.target_variable = target_variable

        # Load dataset (Dask-compatible)
        if file_path.endswith(".nc"):
            self.ds = xr.open_dataset(file_path)
        else:
            self.ds = xr.open_zarr(file_path, consolidated=True, decode_times=False,
                                   storage_options={"token": "anon", "asynchronous": False})
            from cftime import num2date
            units = self.ds.time.attrs.get("units", "hours since 1900-01-01 00:00:0.0")
            cal = self.ds.time.attrs.get("calendar", "standard")
            self.ds["time"] = ("time", num2date(self.ds.time.values, units, calendar=cal))

        # Subset time and region
        self.ds = self.ds.sel(time=time_slice, latitude=slice(*lat_bounds))
        self.ds = self.ds.where((self.ds.longitude >= lon_bounds[0]) | 
                                (self.ds.longitude <= lon_bounds[1]), drop=True)

        # Prepare variable handles (do not load yet)
        self.inputs = {}
        for var in variables:
            da = self.ds[var]
            if "level" in da.dims:
                da = da.mean(dim="level")
            if set(da.dims) == {"latitude", "longitude"}:
                da = da.expand_dims(time=self.ds.time)
            self.inputs[var] = da.chunk({'time': -1})  # Lazy chunking

        self.target = self.ds[target_variable]
        if "level" in self.target.dims:
            self.target = self.target.mean(dim="level")
        self.target = self.target.chunk({'time': -1})

        # Determine sequence count
        self.length = len(self.ds.time) - input_len - forecast_len + 1
        print(f"[Dataset] Initialized with {self.length} samples.")
        print(f"[Dataset] Estimated size: {self.ds.nbytes / 1e6:.2f} MB")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        t0 = idx
        t1 = idx + self.input_len
        tf = t1 + self.forecast_len

        # Stack variables: [C, T, H, W]
        x_vars = []
        for var in self.variables:
            da = self.inputs[var].isel(time=slice(t0, t1))  # [T, H, W]
            x = da.transpose("time", "latitude", "longitude").values  # triggers lazy load
            x_vars.append(x)

        x_array = np.stack(x_vars, axis=0)  # shape: [C, T, H, W]

        # Target: [forecast_len, H, W]
        y = self.target.isel(time=slice(t1, tf)).transpose("time", "latitude", "longitude").values

        return x_array, y
