import xarray as xr
from dask.diagnostics import ProgressBar
with ProgressBar():
    df = xr.open_mfdataset(r'/nesi/nobackup/niwa00004/meyerstj/data/ERA5/CMIP5_downscaling/ICU/*/ERA5_processed_????_???_?.nc')
df = df.sel(time = slice("1980", "2021"))
# there are some issues in the data at the dateline
df = df.sel(longitude = slice(150, 175))
with ProgressBar():
    df = df.load()
df = df.resample(time ='1D').mean()
df1 = df.copy().drop("z")
for varname in ['q','u','v','w','t']:
    for pressure in [500, 850]:
        data = df[varname].sel(pressure = pressure)
        df1[f'{varname}_{pressure}']= (('time','latitude','longitude'), data.values)
    df1 = df1.drop(varname)
    
    
# Something wrong in the commands below

means = df1.mean(["time","latitude","longitude"])
stds = df1.std(["time","latitude","longitude"])

d_means ={}
d_stds = {}
for var in list(df1.data_vars):
    print(var)
    d_means[var] = np.nanmean(df1[var].values)
    d_stds[var] = np.nanstd(df1[var].values)
    
# need to rehash these commands

means.to_netcdf(r'/nesi/project/niwa00018/rampaln/High-res-interpretable-dl/data/era5_means.nc')
stds.to_netcdf(r'/nesi/project/niwa00018/rampaln/High-res-interpretable-dl/data/era5_stds.nc')
df1.to_netcdf(r'/nesi/project/niwa00018/rampaln/High-res-interpretable-dl/data/era5.nc')

