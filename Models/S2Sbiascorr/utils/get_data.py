import xarray as xr
import os
import glob
from pathlib import Path
import re
import datetime
import numpy as np


def fi_preprocess(ds):
    '''
    ---------
    change "time" dimension name to "lead"
    convert "lead" values into 0-45 [days since initialization]
    ---------
    '''
    # init_day = ds.time[0].values
    ds = ds.rename(name_dict={'time':'lead'})
    ds['lead'] = np.arange(46)
    return ds


def calcroll_anom(data):
    # Calculate CLIM
    climatology = data.groupby('init.dayofyear').mean('init')
        
    climCyclical = xr.concat([climatology, climatology, climatology], dim="dayofyear")
    climSmooth0 = climCyclical.rolling(dayofyear=31, center=True).mean()
    climSmooth0 = climSmooth0.rolling(dayofyear=31, center=True).mean()
    climSmooth = climSmooth0.isel(dayofyear=slice(365, 365+365))

    # Calculate ANOMALIES
    anom = data.groupby('init.dayofyear')-climSmooth

    return anom


def calcroll_anomERA5(data):
    # Calculate CLIM
    climatology = data.groupby('time.dayofyear').mean('time')
        
    climCyclical = xr.concat([climatology, climatology, climatology], dim="dayofyear")
    climSmooth0 = climCyclical.rolling(dayofyear=31, center=True).mean()
    climSmooth0 = climSmooth0.rolling(dayofyear=31, center=True).mean()
    climSmooth = climSmooth0.isel(dayofyear=slice(365, 365+365))

    # Calculate ANOMALIES
    anom = data.groupby('time.dayofyear')-climSmooth

    return anom


def get_alldata(ddir):
    root_dir = Path(ddir)
    # have to specify these mems because some init days have more than 11 mems
    pattern = re.compile(r".*(0[0-9]|10)\.nc$")
    
    # grab years 1999-2020 (before real time forecasts)
    files = sorted(str(f) for f in root_dir.glob("*/*/*")
                   if 1999 <= int(f.parent.parent.name) <= 2020 and pattern.match(f.name))
    # split into arrays of 11 members for each init day
    mem_files = np.array_split(files, int(len(files)/11))
    
    da_list = []
    for mem_file in mem_files:
        # open 11 members into one array
        init_da = xr.open_mfdataset(mem_file, preprocess=fi_preprocess, combine='nested', concat_dim='member')
        # convert string date in file name to datetime object
        init_day = datetime.datetime.strptime(mem_file[0][-28:-19], '%d%b%Y')
        # create new dimension named init and assign initialization datetime to it
        init_da = init_da.assign_coords(init=(init_day))
        
        # append 11 members associated with initialization day into list
        da_list.append(init_da)
    
    # Combine all initialization days into one large DataArray
    combined_da = xr.concat(da_list, dim="init")

    return combined_da

def weekmean(data, week, ddir, finame, save=True):
    
    if os.path.exists(ddir+finame):
        print(f"File '{ddir+finame}' exists.")
    
    else:
        weekslice = {
        'init': 0,
        'week12': slice(0, 13),
        'week34': slice(14, 27),
        'week56': slice(28, 41),
        }
        
        time_slice = weekslice.get(week)
    
        # Calculate the member mean, anomalies, then lead week X mean
        da_mean = data.mean('member')
        print("member mean calculated")
        
        da_anom = calcroll_anom(da_mean)
        print("anomalies calculated")

        if week == 'init':
            da_anom_leadavg = da_anom.sel(lead=time_slice)
        else:
            da_anom_leadavg = da_anom.sel(lead=time_slice).mean('lead')
        print("lead week mean calculated")

        da_anom_leadavg.to_netcdf(ddir+finame)
        print('mean saved')

        return da_anom_leadavg
        