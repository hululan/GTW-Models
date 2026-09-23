#######################################
import time

import lightgbm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, cross_validate
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from seaborn import scatterplot
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from tqdm import tqdm
from rasterio.warp import transform
from rasterio.crs import CRS
import xgboost as xgb
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from xgboost import XGBRegressor
import pyproj
import  os
from GTWModel import GTWModel
import warnings
import gc


if __name__ == '__main__':
    print(' 主程序正在启动')
    warnings.filterwarnings(
        "ignore",
        # message="Some inputs do not have OOB scores.*"
    )
    warnings.filterwarnings("ignore", category=FutureWarning)
    data = pd.read_csv('../../data/ChinaPM10/training_NE_China.csv', dtype={'site': str}, parse_dates=['date'])
    # data = data.query('Longitude>90 and Longitude<95 and Latitude>42 and DOY>300 and DOY<360')
    data.dropna(inplace=True)
    # %%

    data.columns
    data.reset_index(drop=True,inplace=True)



    data['date'] = pd.to_datetime(data['date'])
    data['DOY'] = data['date'].dt.strftime('%j').astype('Int16')
    data['day'] = (data['date'] - data['date'].min()).dt.days
    days = data['day'].unique().tolist()
    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(4479)
    # r = pearsonr(data['Fire'],data['PM10'])
    # print(r)

    sites = np.unique(data['site'])

    n, w, s, e = [55, 73, 15, 136]
    nw = [n, w]
    ne = [n, e]
    sw = [s, w]
    se = [s, e]
    center = [(n + s) / 2, (w + e) / 2]
    corners = [nw, ne, sw, se, center]

    for i, corner in enumerate(corners, start=1):
        corner_lat, corner_lon = corner
        lons, lats = data['Longitude'].values, data['Latitude'].values
        corner_lats, corner_lons = np.repeat(corner_lat, len(lons)), np.repeat(corner_lon, len(lats))
        geod = pyproj.Geod(ellps='WGS84')

        _, _, dist = geod.inv(lons, lats, corner_lons, corner_lats)
        dist /= 1000
        data[f'd{i}'] = dist

    selected_cols =['POP','DEM','NDVI','LUC','UW', 'VW', 'DT', 'TEMP', 'BLH', 'SP', 'EVAP', 'PREP','AOD','DOY','RH','WD','WS','Longitude', 'Latitude','d1','d2','d3','d4','d5']
    # selected_cols = ['POP', 'DEM', 'NDVI', 'LUC', 'UW', 'VW', 'DT', 'TEMP', 'BLH', 'SP', 'EVAP', 'PREP', 'AOD',
    #                  'RH', 'WD', 'WS',]
    data['LUC'] = data['LUC'].astype(int).astype('category')
    data['site'] = data['site'].astype('category').cat.codes
    data = data.dropna().copy()
    data.reset_index(drop=True,inplace=True)
    # %%
    from rasterio.crs import CRS
    from rasterio.warp import transform
    # %%
    lon = data['Longitude'].values
    lat = data['Latitude'].values
    # %%
    x,y = transform(src_crs=CRS.from_epsg(4326),dst_crs=CRS.from_epsg(4479),xs = lon,ys = lat)
    # %%
    data['x'] = x
    data['y'] = y
    # %%

    data.dropna(inplace=True)
    data['LUC']=data['LUC'].astype(int).astype('category')

    xy=data[['x','y']]
    data['day'] = (data['date']-data['date'].min()).dt.days
    day = data['day']
    site_id = data['site']
    this_data = data[selected_cols].copy()

    # %%
    this_data = pd.concat([this_data,xy,day,site_id,data[['PM10']]],axis=1)

    kf = KFold(n_splits=10,random_state=42,shuffle=True)
    
    # global_model = ExtraTreesRegressor(n_estimators=50)
    # local_model = ExtraTreesRegressor(n_estimators=50)

    spatial_bandwidths =np.sort(np.unique( np.arange(34,35,5)))
    temporal_bandwidths = np.full_like(spatial_bandwidths,3)
    for spatial_bandwidth,temporal_bandwidth in zip(spatial_bandwidths,temporal_bandwidths):
        print(f'spatial {spatial_bandwidth}, temporal {temporal_bandwidth}')
        s = 'etet'
        params = {
            'global_model': ExtraTreesRegressor(n_estimators=30, max_depth=30, random_state=42),
            'local_model': ExtraTreesRegressor(n_estimators=30, max_depth=30, random_state=42),
            'spatial_bandwidth': spatial_bandwidth,
            'kernel_': 'NN',
            'train_weighted': True,
            'test_weighted': False,
            'temporal_bandwidth': temporal_bandwidth,
            'local_weight': 0.5,
            'nworkers': -1
        }


        a = str(params)

        out_folder = f'../new_training_result_weighted/sp{spatial_bandwidth}tp{temporal_bandwidth}/{s}'
        # if not os.path.exists(out_folder):
        #     continue
        os.makedirs(out_folder,exist_ok=True)
        if os.path.exists(f'{out_folder}/log.txt'):
            pass
        else:

            with open(f'{out_folder}/log.txt', 'w') as f:
                f.write(
                    str(params)
                )
        metrics_list = []
        for i, (train_idx,test_idx) in enumerate(kf.split(this_data),start=1):
            train_set, test_set = this_data.iloc[train_idx,:],this_data.iloc[test_idx,:]
            X_train, X_test = train_set[selected_cols], test_set[selected_cols]
            y_train, y_test = train_set['PM10'], test_set['PM10']
            coords_train, coords_test = train_set[['x','y']].values, test_set[['x','y']].values
            time_train, time_test = train_set['day'].values, test_set['day'].values
            site_train, site_test = train_set['site'].values,test_set['site'].values
            train_set.to_csv(f'{out_folder}/trainSetFold{i}.csv', index=False)
            test_set.to_csv(f'{out_folder}/testSetFold{i}.csv', index=False)
            model = GTWModel(
                **params
            )
            if os.path.exists(f'{out_folder}/resultFold{i}.csv'):
                y_pred = pd.read_csv(f'{out_folder}/resultFold{i}.csv')
                # model = params['global_model']
                # model.fit(X_train,y_train)
                # y_pred['global_predictions'] =model.predict(X_test)
                # local_weight = params['local_weight']
                # y_pred['combined_predictions'] = y_pred['local_predictions']*local_weight + y_pred['global_predictions']*(1-local_weight)
                # y_pred.to_csv(f'{out_folder}/resultFold{i}.csv', index=False)
            else:
                y_pred = model.fit_predict(X_train=X_train,
                                  y_train=y_train,
                                  site_ids_train=site_train,
                                  coords_train=coords_train,
                                  times_train=time_train,
                                  X_test=X_test,
                                  coords_test=coords_test,
                                  times_test=time_test,
                                  )


        # 5. 保存为 CSV 文件
        metrics_df.to_csv(f'{out_folder}/metrics_result.csv', index=False)
