import xarray as xr
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar


def format_features(x_data):
    varnames = list(x_data.data_vars)
    features = xr.concat([x_data[varname] for varname in varnames], dim = "feature")
    # setting a new dimension name in the output
    features['feature'] = (('feature'), varnames)
    features.name = "stacked_features"
    return features
    
def create_test_train_split(config):
    """
    This create a train test validation split of the dataset givena  configuration file
    """
    X = xr.open_dataset(config["X"], chunks = {"time":3000})[config["downscale_variables"]]
    # make sure time values are daily
    X['time'] = pd.to_datetime(X.time.dt.strftime("%Y-%m-%d"))
    y = xr.open_dataset(config["y"], chunks = {"time":3000})
    y['time'] = pd.to_datetime(y.time.dt.strftime("%Y-%m-%d"))
    
    common_times = X.time.to_index().intersection(y.time.to_index())
    X = X.sel(time = common_times)
    y = y.sel(time = common_times)
    # check the y values are also daily
    
    # apply the train test split partiions and load into memory
    with ProgressBar():
        x_train = X.sel(time = slice(config["train_start"], config["train_end"])).load()
        y_train = y.sel(time = slice(config["train_start"], config["train_end"])).load()

        x_test = X.sel(time = slice(config["test_start"], config["test_end"])).load()
        y_test = y.sel(time = slice(config["test_start"], config["test_end"])).load()


        x_val = X.sel(time = slice(config["val_start"], config["val_end"])).load()
        y_val = y.sel(time = slice(config["val_start"], config["val_end"])).load()
                  
    return x_train, x_val, x_test, y_train, y_val, y_test

def prepare_training_dataset(x_train, x_val, x_test,
                             y_train, y_val, y_test,
                             means = None, stds = None):
    """Normalizes and restacks 
    
    the training data so that it is useful for training
    
    """
    if means is None:
        # computing means and stds
        means, stds = x_train.mean(), x_train.std()
    # normalizing the data based on the x_train mean
    x_train_norm = (x_train - means)/stds
    x_test_norm = (x_test- means)/stds
    x_val_norm = (x_val - means)/stds
    try:
        # format the features so that they are stacked
        x_train_norm = format_features(x_train_norm).transpose("time","lat","lon","feature")
        x_test_norm = format_features(x_test_norm).transpose("time","lat","lon","feature")
        x_val_norm = format_features(x_val_norm).transpose("time","lat","lon","feature")
    except ValueError:
        x_train_norm = format_features(x_train_norm).transpose("time","latitude","longitude","feature")
        x_test_norm = format_features(x_test_norm).transpose("time","latitude","longitude","feature")
        x_val_norm = format_features(x_val_norm).transpose("time","latitude","longitude","feature")
    try:
    # prepare the rainfall data
        y_train = y_train.stack(z =['lat','lon']).dropna("z")
        y_test = y_test.stack(z =['lat','lon']).dropna("z")
        y_val= y_val.stack(z =['lat','lon']).dropna("z")
    except ValueError:
        
        y_train = y_train.stack(z =['latitude','longitude']).dropna("z")
        y_test = y_test.stack(z =['latitude','longitude']).dropna("z")
        y_val= y_val.stack(z =['latitude','longitude']).dropna("z")
    
    return x_train_norm, x_test_norm, x_val_norm, y_train, y_test, y_val
    
        
        