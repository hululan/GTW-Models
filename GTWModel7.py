import multiprocessing
import os
from typing import Union
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble._forest import BaseForest
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBModel
from lightgbm.sklearn import LGBMRegressor
from sklearn.utils import resample
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from sklearn.base import clone
from kernel import kernel_function, find_neighbours
import warnings
import gc

warnings.filterwarnings("ignore", category=FutureWarning)




def _construct_weight_worker(args,local_model):
    """
    yielding the spatiotemporal weights for each local model

    :param args: A tuple containing the model parameters and local datasets.
    :param local_model: The local model used to generate local predictions,
        which are combined with the global predictions.
    :return: A tuple containing:
        - **index** (*tuple*): A space-time index in the form ``(time, id_)``.
        - **local_dataset** (*DataFrame or dict*): The local dataset containing
          neighboring observations within the bandwidth of each target observation.
    """

    (time, id_, weighted_row,
     local_X_train, local_y_train,
     st_times_arr, st_site_ids_arr,
     time_slices, neighbours_array,
     bootstrap, train_weighted, resampled,
     kernel_, pow_, temporal_bandwidth) = args
    offset = 1e-6
    sample_weights = []
    av = []
    for this_time in time_slices:
        mask_time = st_times_arr == this_time
        sites_this_time = st_site_ids_arr[mask_time]
        available = np.sort(np.intersect1d(neighbours_array, sites_this_time)).astype(int)
        temporal_kernel = kernel_function(abs(this_time - time),
                                          bandwidth=temporal_bandwidth,
                                          kernel='bisquare' if kernel_ == 'NN' else kernel_,
                                          pow=pow_)
        # if (local_X_test is not None) and (this_time == time):
        #     available = available[available!=id_]
        # NOTE: temporal_bandwidth 通过闭包参数传入
        av.append(available)
        sample_weights.append(weighted_row[available] * temporal_kernel)


    sample_weights = np.concatenate(sample_weights) + offset



    model_local = clone(local_model)
    try:
        n_estimators = model_local.n_estimators
        random_state = model_local.random_state if model_local.random_state else 42
    except:
        n_estimators = 1
        random_state = 42

    local_X = local_X_train
    local_y = local_y_train
    if train_weighted:
        if resampled and len(local_X) < 2 * n_estimators:
            n_extra = min(2 * n_estimators, 2 * len(local_X)) - len(local_X)
            X_extra, y_extra, w_extra = resample(
                local_X, local_y, sample_weights,
                replace=True, n_samples=n_extra, random_state=random_state)
            local_X = pd.concat([local_X, X_extra])
            local_y = pd.concat([local_y, y_extra])
            sample_weights = np.concatenate([sample_weights, w_extra])
        if np.isnan(sample_weights).any():
            print(f'id:{id_}:time{time}, sample weights:{sample_weights}')
            print(f'weight:{weighted_row}, neighbours:{av}')
    else:
        if resampled and len(local_X) < 2 * n_estimators:
            n_extra = min(2 * n_estimators, 2 * len(local_X)) - len(local_X)
            X_extra, y_extra = resample(local_X, local_y,
                                        replace=True, n_samples=n_extra,
                                        random_state=random_state)
            local_X = pd.concat([local_X, X_extra])
            local_y = pd.concat([local_y, y_extra])
        if np.isnan(sample_weights).any():
            print(f'id:{id_}:time{time}, sample weights:{sample_weights}')

    local_ds = pd.concat([local_X, local_y], axis=1,)
    local_ds['weight'] = sample_weights

    return  (time, id_), local_ds

def _match_worker(args):
    """
        Matching the local models with the dataset
        :param args: the parameters and ancillary information for matching local model
         to combine with local predictions
         :return: A dictionary with local dataset and the local model
     """

    (   test_id,
        time_slices,
        time,
        test_weighted,
        this_weight_array,
        this_neighbours_array,
        local_X_test,
        st_times_arr,
        st_site_ids_arr,
        temporal_bandwidth,
        spatial_bandwidth,
        kernel,
        pow_,
    ) = args
    local_map = {}
    local_models_ids:[tuple[int, int]] = []
    weight_parts = []
    offset = 1e-6
    for time_slice in time_slices:
        temporal_kernel = kernel_function(
            np.abs(time_slice - time),
            bandwidth=temporal_bandwidth,
            kernel='bisquare' if kernel == 'NN' else kernel,
            pow=pow_)
        mask = st_times_arr == time_slice
        site_this_time = np.sort(st_site_ids_arr[mask])
        available = np.sort(np.intersect1d(site_this_time, this_neighbours_array)).astype(int)
        w = this_weight_array[available] * temporal_kernel
        weight_parts.append(w)
        model_ids_this_time = [(time_slice, site_id) for site_id in available.tolist()]
        local_models_ids.extend(model_ids_this_time)
    local_model_weights = np.concatenate(weight_parts) + offset
    n=15
    if test_weighted:
        local_model_weights = local_model_weights / local_model_weights.sum()
        for key, weight in zip(local_models_ids, local_model_weights):
            local_X_test['weight'] = weight
            local_X_test['id'] = test_id
            local_map[key] = local_X_test.copy()
    else:
        topn_idx = np.argpartition(local_model_weights, -n)[-n:] if len(local_model_weights) > n else np.arange(0, len(local_model_weights))
        local_model_weights = local_model_weights[topn_idx]/local_model_weights[topn_idx].sum()
        for idx, weight in zip(list(topn_idx), local_model_weights):
            key = local_models_ids[idx]
            local_X_test['weight'] = weight
            local_X_test['id'] = test_id
            local_map[key] = local_X_test.copy()

        # nearest_idx = int(np.argmax(local_model_weights))
        # key = local_models_ids[nearest_idx]
        # local_X_test['weight'] = 1
        # local_X_test['id'] = test_id
        # local_map[key] = local_X_test.copy()
        # if max(local_model_weights) > threshold:
        #     nearest_idx = int(np.argmax(local_model_weights))
        #     key = local_models_ids[nearest_idx]
        #     local_X_test['weight'] = 1
        #     local_X_test['id'] = test_id
        #     local_map[key] = local_X_test.copy()
        # else:
        #     idx = np.argsort(local_model_weights)[::-1]
        #     weights_sorted = local_model_weights[idx]
        #     cum = np.cumsum(weights_sorted)
        #     n = np.searchsorted(cum,threshold)+1
        #     selected_idx = idx[:n]
        #     selected_weights = local_model_weights[selected_idx]
        #     selected_weights /= selected_weights.sum()
        #     for idx_, weight in zip(selected_idx.tolist(), local_model_weights):
        #         key = local_models_ids[idx_]
        #         local_X_test['weight'] = weight
        #         local_X_test['id'] = test_id
        #         local_map[key] = local_X_test.copy()



    return local_map

def cleanup(item):
    _,obj = item
    obj.__del__() if hasattr(obj,'__del__') else None

def _predict_tuple_worker(args):
    """predict using local models and weights
    """
    (local_model,
     local_X_test) =args
    ids = local_X_test['id']
    weights = local_X_test['weight']
    local_X_test = local_X_test.drop(['weight','id'], axis=1)
    local_preds =local_model.predict(local_X_test)
    local_preds = local_preds*weights
    local_result = pd.DataFrame({
        'id':ids,
        'y_pred':local_preds,
    })
    return local_result

def model_fit_predit(args, model):
    """
         fitting and predicting local models with spatiotemporal weights
         :param args: idx indicating the index of the model, local_ds indicating the dataset
         :param model: local model generating local predictions to combine with global predictions
         :return: idx, fitted model
         A tuple containing:
        - **local_dataset** (*DataFrame*): the local predictions of the given local dataset.
     """
    (
        id_,
        local_ds,
        X_test,
    ) = args
    _, model = model_fitting((id_,local_ds),model)
    return _predict_tuple_worker((model, X_test))

def model_fit_importance(args, model):
    """
         fitting local models with spatiotemporal weights, and yielding the local importance
         :param args: idx indicating the index of the model, local_ds indicating the dataset
         :param model: local model generating local predictions to combine with global predictions
         :return: idx, fitted model
         A tuple containing:
        - **index** (*tuple*): A space-time index in the form ``(time, id_)``.
        - **importance** : importance from fitted model.
    """
    (
        id_,
        local_ds
    ) = args
    _, model = model_fitting((id_,local_ds),model)
    return id_, model.feature_importances_

def model_fitting(args,model):
    """
         fitting local models with spatiotemporal weights
         :param args: idx indicating the index of the model, local_ds indicating the dataset
         :param model: local model generating local predictions to combine with global predictions
                  A tuple containing:
        - **index** (*tuple*): A space-time index in the form ``(time, id_)``.
        - **fitted model** : The fitted local model.
     """
    (idx,local_ds) = args
    X_train = local_ds.drop(columns=['weight','y'])
    weights = local_ds['weight']
    y_train=local_ds['y']
    model.fit(X_train, y_train, weights)
    return idx, model


def concat_worker(item):
    key, data = item
    if len(data) >0:
        return key, {'train': None, 'test': pd.concat(data, ignore_index=True)}
    else:
        return None, {'train': None, 'test': None}


class GTWModel:
    def __init__(self,
                 global_model:Union[BaseForest, LGBMRegressor,XGBModel,DecisionTreeRegressor],
                 local_model:Union[BaseForest, LGBMRegressor,XGBModel,DecisionTreeRegressor],
                 spatial_bandwidth: float,
                 temporal_bandwidth: float,
                 train_weighted: bool = True,
                 test_weighted: bool = False,
                 kernel_: str = 'bisquare',
                 bootstrap: bool = True,
                 nworkers: int = 1,          # ← 默认改为 4，充分利用 CPU
                 resampled: bool = False,
                 local_weight = .5,
                 rf_add_params: dict = None):
        """
            The framework of geographically and temporally weighted models, training local models on each observation to tackling spatiotemporal non-stationarity.
            :param global_model: the global models generating global predictions to combine with local predictions
            :param local_model: the local models generating local predictions to combine with global predictions
            :param spatial_bandwidth: the spatial bandwidth (how much spatial neighboring observations are accounted while fitting local models)
            :param temporal_bandwidth: the temporal bandwidth (how much temporal neighboring observations are accounted while fitting local models)
            :param kernel_: kernel type including 'bisquare','gaussian','inverse','normal','NN'
            :param nworkers: numbers of parallel workers, -1 is all the cpus
            :param local_weight: the weight of local predictions while combining with global predictions
        """
        if rf_add_params is None:
            rf_add_params = {}
        self.global_model = global_model
        self.local_model = local_model
        self.spatial_bandwidth = spatial_bandwidth
        self.temporal_bandwidth = temporal_bandwidth
        self.bootstrap = bootstrap
        self.train_weighted = train_weighted
        self.test_weighted = test_weighted
        self.kernel_ = kernel_
        self.resampled = resampled
        self.model_map = {

        }
        self.unique_attrs = None
        self.pow = 1
        self.local_weight = local_weight
        if nworkers >= 1:
            self.workers = nworkers
        else:
            self.workers = multiprocessing.cpu_count()
        # ── 预计算缓存 ──────────────────────────────────────────────
        self._st_times_arr: NDArray = None    # ST_attrs['time'] 的 numpy 副本
        self._st_site_ids_arr: NDArray = None # ST_attrs['site_id'] 的 numpy 副本
        if self.local_weight > 1:
            raise ValueError(f'local_weight 不应大于 1，当前值：{self.local_weight}')


    # ── fit ────────────────────────────────────────────────────────────────
    def fit(self,
            X_train: Union[NDArray, pd.DataFrame],
            y_train: Union[NDArray, pd.Series],
            site_ids: Union[NDArray, pd.Series],
            coords: Union[NDArray, pd.DataFrame],
            times: Union[NDArray, pd.Series]):
        """
            fitting models with training set and spatiotemporal information
            :param X_train: the DataFrame or matrix containing training set
            :param y_train: the series or array containing the labels or true values
            :param site_ids: the ids with unique spatial coordinates
            :param coords: the DataFrame or matrix containing spatial coordinates
            :param times: the series or array containing the times
            :return: self
        """

        X_train = X_train.reset_index(drop=True) if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
        y_train = y_train.reset_index(drop=True) if isinstance(y_train, pd.Series) else pd.Series(y_train)
        self.global_model.n_jobs = self.workers
        self.global_model.fit(X_train, y_train)
        print('global_model fit complete')

        # ST 属性表
        coords_arr = np.asarray(coords)
        times_arr = np.asarray(times)
        site_ids = pd.Series(site_ids).astype('category').cat.codes
        site_id_arr = np.asarray(site_ids)


        self._st_times_arr = times_arr
        self._st_site_ids_arr = site_id_arr

        global_data_set = X_train.copy() if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
        global_data_set['y'] = np.asarray(y_train)
        global_data_set['site_id'] = site_id_arr
        global_data_set['time'] = times_arr

        unique_attrs = (pd.DataFrame({'site_id': site_id_arr,
                                      'x': coords_arr[:, 0],
                                      'y': coords_arr[:, 1]})
                        .groupby('site_id')[['x', 'y']].first()
                        .reset_index()
                        .sort_values('site_id'))
        self.unique_attrs = unique_attrs
        unique_coords = unique_attrs[['x', 'y']].values

        dist_matrix = (cdist(unique_coords, unique_coords, metric='euclidean')
                       if self.train_weighted
                       else np.ones((len(unique_coords), len(unique_coords))))

        weighted = kernel_function(distances=dist_matrix,
                                   bandwidth=self.spatial_bandwidth,
                                   kernel=self.kernel_, pow=self.pow)
        neighbours = find_neighbours(distances=dist_matrix,
                                     bandwidth=self.spatial_bandwidth,
                                     kernel=self.kernel_)


        local_data_map = self.model_map.copy()
        n_total = len(site_id_arr)

        # ▶ 优化⑤：多进程并行训练局部模型
        def iter_task_args():
            for time_val,id_ in zip(times_arr,site_id_arr):

                neighbours_array, = np.where(neighbours[id_, :])

                con = np.logical_and(
                        np.abs(self._st_times_arr - time_val) < self.temporal_bandwidth,
                        np.isin(self._st_site_ids_arr, neighbours_array)
                )

                local_ds = global_data_set[con].sort_values(['time', 'site_id'])
                local_X = local_ds.drop(columns=['y', 'time', 'site_id'])
                local_y = local_ds['y']
                time_slices = np.sort(np.unique(self._st_times_arr[con]))

                yield (
                    time_val, id_,
                    weighted[id_, :],
                    local_X, local_y,
                    self._st_times_arr, self._st_site_ids_arr,
                    time_slices, neighbours_array,
                    self.bootstrap, self.train_weighted, self.resampled,
                    self.kernel_, self.pow,  self.temporal_bandwidth
                )

                # 显式释放当前循环里的临时对象
                del local_ds, local_X, local_y, time_slices, neighbours_array, con

        worker = partial(_construct_weight_worker,
                         local_model = self.local_model)

        with Pool(processes=self.workers) as pool:
            for idx, local_train_data in tqdm(
                    pool.imap_unordered(
                        worker,
                        iter_task_args(),
                        chunksize=1
                    ),
                    total=n_total,
                    desc='matching data and weight'
            ):
                local_data_map[idx] = local_train_data

        worker = partial(model_fitting,
                         model = self.local_model)

        with Pool(processes=self.workers) as pool:
            for idx, local_model in tqdm(
                    pool.imap_unordered(
                        worker,
                        local_data_map.items(),
                        chunksize=1
                    ),
                    total=n_total,
                    desc='fitting local models'
            ):
                self.model_map[idx] = local_model

        return self

    # ── predict ─────────────────────────────────────────────────────────────
    def predict(self,
                X_test: Union[pd.DataFrame, NDArray],
                coords: Union[pd.DataFrame, NDArray],
                times: Union[pd.Series, NDArray]):
        """
            predicting data with testing data together with spatiotemporal information
            :param X_test: the DataFrame or matrix containing testing data
            :param coords: the DataFrame or matrix containing spatial coordinates
            :param times: the series or array containing the times
            :return: predictions in series
        """

        local_weight = self.local_weight
        X_test = X_test.reset_index(drop=True) if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)

        unique_test_coords, test_site_ids  = np.unique(coords,axis=0, return_inverse=True)

        test_dist_matrix = cdist(unique_test_coords, self.unique_attrs[['x', 'y']].values, 'euclidean')
        neighbours_matrix = find_neighbours(test_dist_matrix, self.spatial_bandwidth, self.kernel_)
        test_spatial_weight_matrix = kernel_function(
            test_dist_matrix, bandwidth=self.spatial_bandwidth,
            kernel=self.kernel_, pow=self.pow)

        global_predictions = self.global_model.predict(X_test)

        local_data_map = {}

        for key in self.model_map.keys():
            local_data_map[key] = []

        def generate_local_data():
            for test_id, (time,site_id) in enumerate(zip(times,test_site_ids)):
                this_weight_array = test_spatial_weight_matrix[site_id, :]
                this_neighbours_array, = np.where(neighbours_matrix[site_id, :])
                local_X_test = X_test.iloc[test_id: test_id + 1, :].copy()
                model_idx = np.abs(self._st_times_arr - time) < self.temporal_bandwidth
                time_slices = np.unique(self._st_times_arr[model_idx])
                yield(
                    test_id,
                    time_slices,
                    time,
                    self.test_weighted,
                    this_weight_array,
                    this_neighbours_array,
                    local_X_test,
                    self._st_times_arr,
                    self._st_site_ids_arr,
                    self.temporal_bandwidth,
                    self.spatial_bandwidth,
                    self.kernel_,
                    self.pow,
                )
                del time_slices,this_weight_array,this_neighbours_array,local_X_test
        with Pool(self.workers) as pool:
            for local_map in tqdm(pool.imap_unordered(
                        _match_worker,
                        generate_local_data(),
                                ),total=X_test.shape[0],desc='matching models and local data'):
                for key, value in local_map.items():
                    local_data_map[key].append(value)


        local_data_map = {
            key: pd.concat(data, ignore_index=True)
            for key, data in local_data_map.items()
            if len(data) > 0
        }


        def iter_task_args():
            for key in local_data_map:
                local_data = local_data_map[key]
                local_model = self.model_map[key]
                yield (local_model, local_data)
                del local_data, local_model,key

        local_predictions = []
        with Pool(self.workers) as pool:
            for prediction in tqdm(pool.imap_unordered(
                        _predict_tuple_worker,
                        iter_task_args(),
                                ),total=len(local_data_map),desc='predicting'):
                local_predictions.append(prediction)
        local_predictions = pd.concat(local_predictions,ignore_index=True)
        predictions = local_predictions.groupby(['id']).sum()
        predictions.sort_index(ascending=True, inplace=True)
        predictions.rename(columns={'y_pred': 'local_predictions'}, inplace=True)

        predictions['global_predictions'] = global_predictions
        predictions['combined_predictions'] = predictions['local_predictions']*local_weight + predictions['global_predictions']*(1-local_weight)
        return predictions

    # ── 辅助信息提取 ────────────────────────────────────────────────────────
    def get_local_feature_importance(self):
        """getting global and local feature importances
            :return: a DataFrame containing global and local feature importances"""
        if self.model_map is None:
            print("模型尚未训练")
            return None
        global_feature_importance = [['global']+list(self.global_model.feature_importances_)]
        rows = [[i] + list(m.feature_importances_) for i, m in tqdm(enumerate(self.model_map.values()),total=len(self.model_map))]
        cols = ['model_index'] + [f'f{i}' for i in range(len(rows[0]) - 1)]
        return pd.DataFrame(global_feature_importance + rows, columns=cols)

    # def get_local_R2(self):
    #     if self.model_map is None:
    #         print("模型尚未训练")
    #         return None
    #     rows = [[i, m.oob_score_] for i, m in enumerate(self.model_map.values())]
    #     return pd.DataFrame(rows, columns=['model_index', 'local R2'])
    # def clear(self):
    #     with Pool(self.workers) as pool:
    #         list(
    #             tqdm(
    #                 pool.imap_unordered(
    #                     cleanup,
    #                     self.model_map.items()),
    #                 total=len(self.model_map),
    #                 desc='clearing models')
    #         )
    #     gc.collect()

    def station_cv(self,
            X: Union[NDArray, pd.DataFrame],
            y: Union[NDArray, pd.Series],
            site_ids: Union[NDArray, pd.Series],
            coords: Union[NDArray, pd.DataFrame],
            times: Union[NDArray, pd.Series]):
        """ station cross validation: leaving one station out for validating, using the others stations for model fitting,
            :param X: the DataFrame or matrix containing all explanatory variable
            :param y: the series or array containing the labels or true values
            :param site_ids: the ids with unique spatial coordinates
            :param coords: the DataFrame or matrix containing spatial coordinates
            :param times: the series or array containing the times
            :return: a DataFrame containing global and local feature importances"""
        self.global_model.n_jobs = self.workers
        # 全局模型
        X = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = y.reset_index(drop=True) if isinstance(y, pd.Series) else pd.Series(y)
        site_ids = site_ids.reset_index(drop=True) if isinstance(site_ids, pd.Series) else pd.Series(site_ids)
        coords = coords.reset_index(drop=True) if isinstance(coords, pd.DataFrame) else pd.DataFrame(coords)
        times = times.reset_index(drop=True) if isinstance(times, pd.Series) else pd.Series(times)
        results = []
        unique_site_ids = np.unique(site_ids)
        for site_id in unique_site_ids:
            print(f'{site_id}/ {len(unique_site_ids)}')
            con = site_ids!=site_id
            X_train, X_test = X[con], X[~con]
            y_train, y_test = y[con], y[~con]
            coords_train, coords_test = coords[con], coords[~con]
            times_train, times_test = times[con], times[~con]
            site_ids_train, site_ids_test = site_ids[con], site_ids[~con]
            args = (X_train, y_train, site_ids_train, coords_train, times_train,X_test, coords_test, times_test)
            result = self.fit_predict(*args)
            result['true'] = y_test.values
            result['site_id'] = site_id
            results.append(result)

        return pd.concat(results)
    def time_cv(self,
            X: Union[NDArray, pd.DataFrame],
            y: Union[NDArray, pd.Series],
            site_ids: Union[NDArray, pd.Series],
            coords: Union[NDArray, pd.DataFrame],
            times: Union[NDArray, pd.Series]):
        """
            time cross validation: leaving one station out for validating, using the others stations for model fitting,
            :param X: the DataFrame or matrix containing all explanatory variable
            :param y: the series or array containing the labels or true values
            :param site_ids: the ids with unique spatial coordinates
            :param coords: the DataFrame or matrix containing spatial coordinates
            :param times: the series or array containing the times
            :return: a DataFrame containing global and local feature importances
        """
        self.global_model.n_jobs = self.workers
        # 全局模型
        X = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = y.reset_index(drop=True) if isinstance(y, pd.Series) else pd.Series(y)
        site_ids = site_ids.reset_index(drop=True) if isinstance(site_ids, pd.Series) else pd.Series(site_ids)
        coords = coords.reset_index(drop=True) if isinstance(coords, pd.DataFrame) else pd.DataFrame(coords)
        times = times.reset_index(drop=True) if isinstance(times, pd.Series) else pd.Series(times)
        results = []
        unique_times = np.unique(times)
        for this_time in unique_times:
            print(f'{this_time}/ {len(unique_times)}')
            con = times!=this_time
            X_train, X_test = X[con], X[~con]
            y_train, y_test = y[con], y[~con]
            coords_train, coords_test = coords[con], coords[~con]
            times_train, times_test = times[con], times[~con]
            site_ids_train, site_ids_test = site_ids[con], site_ids[~con]
            args = (X_train, y_train, site_ids_train, coords_train, times_train,X_test, coords_test, times_test)
            result = self.fit_predict(*args)
            result['true'] = y_test.values
            result['time'] = this_time
            results.append(result)

        return pd.concat(results)

    def sample_cv(self,
            X: Union[NDArray, pd.DataFrame],
            y: Union[NDArray, pd.Series],
            site_ids: Union[NDArray, pd.Series],
            coords: Union[NDArray, pd.DataFrame],
            times: Union[NDArray, pd.Series]):
        """
            time cross validation: leaving one station out for validating, using the others stations for model fitting,
            :param X: the DataFrame or matrix containing all explanatory variable
            :param y: the series or array containing the labels or true values
            :param site_ids: the ids with unique spatial coordinates
            :param coords: the DataFrame or matrix containing spatial coordinates
            :param times: the series or array containing the times
            :return: a DataFrame containing global and local feature importances
        """
        X = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = y.reset_index(drop=True) if isinstance(y, pd.Series) else pd.Series(y)
        final_results = pd.DataFrame({
            'time' :times,
            'site_id': site_ids,
        })
        global_model_worker = partial(model_fit_predit,self.global_model)
        def generate_global_data():
            for  i in range(0, X.shape[0]):
                con = np.ones(X.shape[0], dtype=bool)
                con[i] = False
                X_train, X_test = X[con], X[~con]
                y_train = y[con]
                yield  i, X_train, y_train, np.ones_like(y_train), X_test
                del X_train, X_test, y_train
        results = []
        with Pool(self.workers) as pool:
            for result in tqdm(
                pool.imap_unordered(
                    model_fit_predit,
                    generate_global_data()
                ),
                total=X.shape[0],
                desc='global_model fitting and predicting',
            ):
                results.append(result)
        # for args in tqdm(generate_global_data(),total=X.shape[0]):
        #     global_model = args[0]
        #     global_model.n_jobs = self.workers
        #     global_model.fit(args[2], args[3],args[4])
        #     results.append(global_model.predict(args[5]))

        global_y_preds = pd.concat(results)
        final_results['global_predictions'] =global_y_preds['y_pred'].values

        local_data_map = self.sample_cv_matching(X,y, site_ids, coords, times)
        def generate_local_data():
            model = clone(self.local_model)
            for i,key in enumerate(zip(times,site_ids)):
                X_train =local_data_map[key]['train'].drop(columns=['y', 'weight'])
                y_train = local_data_map[key]['train']['y']
                weights = local_data_map[key]['train']['weight']
                X_test = local_data_map[key]['test']
                yield  model ,i, X_train.copy(), y_train.copy(), weights, X_test.copy()
                del X_train, X_test, y_train
        results = []
        with Pool(self.workers) as pool:
            for result in tqdm(
                pool.imap_unordered(
                    model_fit_predit,
                    generate_local_data()
                ),
                total = len(times),
                desc='local_model fitting and predicting',
            ):
                results.append(result)
        local_y_preds = pd.concat(results).sort_values(by = ['id'])
        final_results['local_predictions'] = local_y_preds['y_pred'].values

        local_weights = self.local_weight
        final_results['combined_predictions'] =final_results['global_predictions']*(1-local_weights)+ final_results['local_predictions']*local_weights
        final_results['true'] = y.values

        return final_results


    def fit_predict(self,
                    X_train: Union[NDArray, pd.DataFrame],
                    y_train: Union[NDArray, pd.DataFrame],
                    site_ids_train: Union[NDArray, pd.Series],
                    coords_train: Union[NDArray, pd.DataFrame],
                    times_train: Union[NDArray, pd.Series],
                    X_test: Union[NDArray, pd.DataFrame],
                    coords_test: Union[NDArray, pd.DataFrame],
                    times_test: Union[NDArray, pd.Series],
                    concat_data=None):
        """
            using constructing and discarding strategy to overcome the problem storing local models, which is memory-intensive
            :param X_train: the DataFrame or matrix containing training explanatory variable
            :param y_train: the training series or array containing the labels or true values
            :param site_ids_train: the ids with unique spatial coordinates
            :param coords_train: the DataFrame or matrix containing spatial coordinates
            :param times_train: the series or array containing the times
            :param X_test: the DataFrame or matrix containing testing explanatory variable
            :param site_ids_test: the ids with unique spatial coordinates
            :param coords_test: the DataFrame or matrix containing spatial coordinates
            :param times_test: the series or array containing the times
            :return:
        """
        self.global_model.n_jobs = self.workers
        self.global_model.fit(X_train, y_train)
        print('global_model fit complete')
        global_predictions = self.global_model.predict(X_test)
        local_weight = self.local_weight
        # ST 属性表
        coords_arr = np.asarray(coords_train)
        times_arr = np.asarray(times_train)
        site_ids_train = pd.Series(site_ids_train).astype('category').cat.codes
        site_id_arr = np.asarray(site_ids_train)


        self._st_times_arr = times_arr
        self._st_site_ids_arr = site_id_arr

        global_data_set = X_train.copy() if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
        global_data_set['y'] = np.asarray(y_train)
        global_data_set['site_id'] = site_id_arr
        global_data_set['time'] = times_arr

        unique_attrs = (pd.DataFrame({'site_id': site_id_arr,
                                      'x': coords_arr[:, 0],
                                      'y': coords_arr[:, 1]})
                        .groupby('site_id')[['x', 'y']].first()
                        .reset_index()
                        .sort_values('site_id'))



        self.unique_attrs = unique_attrs
        unique_coords = unique_attrs[['x', 'y']].values
        unique_test_coords, test_site_ids  = np.unique(coords_test,axis=0, return_inverse=True)

        test_distance_array = (cdist(unique_test_coords, unique_coords, metric='euclidean')
                               if self.train_weighted
                               else np.ones((len(unique_test_coords), len(unique_coords))))

        test_weight_array = kernel_function(
            test_distance_array,
            bandwidth=self.spatial_bandwidth,
            kernel=self.kernel_,
            pow=self.pow
        )
        test_neighbours = find_neighbours(distances= test_distance_array,
                                     bandwidth=self.spatial_bandwidth,
                                     kernel=self.kernel_)

        local_data_map_1 = {
            (time,site_id):[]
            for time, site_id in zip(times_train, site_ids_train)
        }

        def generate_local_data():
            for test_id, (time,site_id) in enumerate(zip(times_test, test_site_ids)):
                this_weight_array = test_weight_array[site_id,:].copy()
                this_neighbours_array, =  np.where(test_neighbours[site_id,:])
                local_X_test = X_test.iloc[test_id: test_id + 1, :].copy()
                model_idx = np.abs(self._st_times_arr - time) < self.temporal_bandwidth
                time_slices = np.unique(self._st_times_arr[model_idx])
                yield (test_id,
                       time_slices,
                       time,
                       self.test_weighted,
                       this_weight_array,
                       this_neighbours_array,
                       local_X_test,
                       self._st_times_arr.copy(),
                       self._st_site_ids_arr.copy(),
                       self.temporal_bandwidth,
                       self.spatial_bandwidth,
                       self.kernel_,
                       self.pow,
                       )
                del time_slices, this_weight_array, this_neighbours_array, local_X_test

        with Pool(self.workers) as pool:
            for local_map in tqdm(pool.imap_unordered(
                    _match_worker,
                    generate_local_data(),
            ), total=X_test.shape[0], desc='matching models and local data'):
                for key,value in local_map.items():
                    local_data_map_1[key].append(value)

        local_data_map ={}
        with Pool(self.workers) as pool:
            for key, concat_result in tqdm(pool.imap_unordered(concat_worker,local_data_map_1.items()), total = len(local_data_map_1), desc='concatenating local data'):
                if key is not None:
                    local_data_map[key]=concat_result
        del local_data_map_1
        # local_data_map = {
        #     key: pd.concat(data, ignore_index=True)
        #     for key, data in tqdm(local_data_map.items(),total=len(local_data_map), desc='concatenating local data')
        #     if len(data) > 0
        # }

        #计算距离矩阵，权重矩阵和近邻矩阵
        dist_matrix = (cdist(unique_coords, unique_coords, metric='euclidean')
                       if self.train_weighted
                       else np.ones((len(unique_coords), len(unique_coords))))

        weighted = kernel_function(distances=dist_matrix,
                                   bandwidth=self.spatial_bandwidth,
                                   kernel=self.kernel_, pow=self.pow)
        neighbours = find_neighbours(distances=dist_matrix,
                                     bandwidth=self.spatial_bandwidth,
                                     kernel=self.kernel_)


        # ▶ 优化⑤：多进程并行训练局部模型
        def iter_task_args():
            for time_val,id_ in local_data_map.keys():
                neighbours_array, = np.where(neighbours[id_, :])

                con = np.logical_and(
                        np.abs(self._st_times_arr - time_val) < self.temporal_bandwidth,
                        np.isin(self._st_site_ids_arr, neighbours_array)
                )

                local_ds = global_data_set[con].sort_values(['time', 'site_id']).copy()
                local_X = local_ds.drop(columns=['y', 'time', 'site_id']).copy()
                local_y = local_ds['y'].copy()
                time_slices = np.sort(np.unique(self._st_times_arr[con]))

                yield (
                    time_val, id_,
                    weighted[id_, :],
                    local_X, local_y,
                    self._st_times_arr, self._st_site_ids_arr,
                    time_slices, neighbours_array,
                    self.bootstrap, self.train_weighted, self.resampled,
                    self.kernel_, self.pow,  self.temporal_bandwidth
                )

                # 显式释放当前循环里的临时对象
                del local_ds, local_X, local_y, time_slices, neighbours_array, con

        worker = partial(_construct_weight_worker,
                         local_model = self.local_model)

        with Pool(processes=self.workers) as pool:
            for key, local_train_data in tqdm(
                    pool.imap_unordered(
                        worker,
                        iter_task_args(),
                        chunksize=1
                    ),
                    total=len(local_data_map),
                    desc='matching data and weight'
            ):
                local_data_map[key]['train'] = local_train_data.copy()

        def generate_fit_predtict():
            for key in local_data_map.keys():
                yield (
                    0,
                    local_data_map[key]['train'],
                    local_data_map[key]['test'],

                )
        local_predictions = []
        fit_predtict_worker = partial(model_fit_predit,model = self.local_model)
        with Pool(processes=self.workers) as pool:
            for result in tqdm(pool.imap_unordered(
                fit_predtict_worker,
                generate_fit_predtict()),
                total=len(local_data_map),
                desc='predicting data'
            ):
                local_predictions.append(result)

        local_predictions =pd.concat(local_predictions, ignore_index=True)
        predictions = local_predictions.groupby(['id']).sum()
        a = np.unique(local_predictions['id'])
        b = np.arange(a.min(), a.max() + 1)
        c = b[~np.isin(b,a)]
        predictions.sort_index(ascending=True, inplace=True)
        predictions.rename(columns={'y_pred': 'local_predictions'}, inplace=True)
        predictions['global_predictions'] = global_predictions
        predictions['combined_predictions'] = (predictions['local_predictions'] * local_weight +
                                               predictions['global_predictions'] * (1 - local_weight))
        return predictions


    def sample_cv_matching(self,
        X: Union[NDArray, pd.DataFrame],
        y: Union[NDArray, pd.Series],
        site_ids: Union[NDArray, pd.Series],
        coords: Union[NDArray, pd.DataFrame],
        times: Union[NDArray, pd.Series]):
        """
            leave one sample out,
            :param X: the DataFrame or matrix containing training explanatory variables
            :param y: the training series or array containing the labels or true values
            :param site_ids: the ids with unique spatial coordinates
            :param coords: the DataFrame or matrix containing spatial coordinates
            :param times: the series or array containing the times
            :return: a DataFrame of predictions
        """

        coords_arr = np.asarray(coords)
        times_arr = np.asarray(times)
        site_ids = pd.Series(site_ids).astype('category').cat.codes
        site_id_arr = np.asarray(site_ids)

        self._st_times_arr = times_arr
        self._st_site_ids_arr = site_id_arr

        global_data_set = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        global_data_set['y'] = np.asarray(y)
        global_data_set['site_id'] = site_id_arr
        global_data_set['time'] = times_arr

        unique_attrs = (pd.DataFrame({'site_id': site_id_arr,
                                      'x': coords_arr[:, 0],
                                      'y': coords_arr[:, 1]})
                        .groupby('site_id')[['x', 'y']].first()
                        .reset_index()
                        .sort_values('site_id'))

        unique_coords = unique_attrs[['x', 'y']].values

        # 计算距离矩阵，权重矩阵和近邻矩阵
        dist_matrix = (cdist(unique_coords, unique_coords, metric='euclidean')
                       if self.train_weighted
                       else np.ones((len(unique_coords), len(unique_coords))))

        weighted = kernel_function(distances=dist_matrix,
                                   bandwidth=self.spatial_bandwidth,
                                   kernel=self.kernel_, pow=self.pow)
        neighbours = find_neighbours(distances=dist_matrix,
                                     bandwidth=self.spatial_bandwidth,
                                     kernel=self.kernel_)

        result = {
            (time,site): {}
            for time,site
            in zip(self._st_times_arr,self._st_site_ids_arr)
        }

        def iter_task_args():
            for time_val, id_ in result.keys():
                neighbours_array, = np.where(neighbours[id_, :])

                con = np.logical_and(
                    np.abs(self._st_times_arr - time_val) < self.temporal_bandwidth,
                    np.isin(self._st_site_ids_arr, neighbours_array)
                )

                local_ds = global_data_set[con].sort_values(['time', 'site_id'])
                exception = np.logical_and(
                    local_ds['site_id'] == id_,
                    local_ds['time'] == time_val
                )
                local_ds = local_ds[~exception]
                local_X_train = local_ds.drop(columns=['y', 'time', 'site_id'])
                local_y_train = local_ds['y']
                time_slices = np.sort(np.unique(self._st_times_arr[con]))


                yield (
                    time_val, id_,
                    weighted[id_, :],
                    local_X_train, local_y_train,
                    self._st_times_arr, self._st_site_ids_arr,
                    time_slices, neighbours_array,
                    self.bootstrap, self.train_weighted, self.resampled,
                    self.kernel_, self.pow, self.temporal_bandwidth,
                )

                # 显式释放当前循环里的临时对象
                del local_ds, local_X_train, local_y_train, time_slices, neighbours_array, con

        worker = partial(_construct_weight_worker,
                         local_model=self.local_model)

        with Pool(processes=self.workers) as pool:
            for idx,local_ds_train in tqdm(
                    pool.imap_unordered(
                        worker,
                        iter_task_args(),
                        chunksize=1
                    ),
                    total=len(result),
                    desc='matching data and weight'
            ):
                result[idx]['train'] = local_ds_train

        return result

    def fit_importance(self,
            X_train: Union[NDArray, pd.DataFrame],
            y_train: Union[NDArray, pd.Series],
            site_ids: Union[NDArray, pd.Series],
            coords: Union[NDArray, pd.DataFrame],
            times: Union[NDArray, pd.Series]):
        """
            fitting models and yielding the importances of each features
            :param X_train: the DataFrame or matrix containing training explanatory variable
            :param y_train: the training series or array containing the labels or true values
            :param site_ids: the ids with unique spatial coordinates
            :param coords: the DataFrame or matrix containing spatial coordinates
            :param times: the series or array containing the times
            :return: a DataFrame containing global and local feature importances
        """

        # 全局模型
        X_train = X_train.reset_index(drop=True) if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
        y_train = y_train.reset_index(drop=True) if isinstance(y_train, pd.Series) else pd.Series(y_train)
        self.global_model.n_jobs = self.workers
        self.global_model.fit(X_train, y_train)
        print('global_model fit complete')

        # ST 属性表
        coords_arr = np.asarray(coords)
        times_arr = np.asarray(times)
        site_ids = pd.Series(site_ids).astype('category').cat.codes
        site_id_arr = np.asarray(site_ids)


        self._st_times_arr = times_arr
        self._st_site_ids_arr = site_id_arr

        global_data_set = X_train.copy() if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
        global_data_set['y'] = np.asarray(y_train)
        global_data_set['site_id'] = site_id_arr
        global_data_set['time'] = times_arr

        unique_attrs = (pd.DataFrame({'site_id': site_id_arr,
                                      'x': coords_arr[:, 0],
                                      'y': coords_arr[:, 1]})
                        .groupby('site_id')[['x', 'y']].first()
                        .reset_index()
                        .sort_values('site_id'))
        self.unique_attrs = unique_attrs
        unique_coords = unique_attrs[['x', 'y']].values

        dist_matrix = (cdist(unique_coords, unique_coords, metric='euclidean')
                       if self.train_weighted
                       else np.ones((len(unique_coords), len(unique_coords))))

        weighted = kernel_function(distances=dist_matrix,
                                   bandwidth=self.spatial_bandwidth,
                                   kernel=self.kernel_, pow=self.pow)
        neighbours = find_neighbours(distances=dist_matrix,
                                     bandwidth=self.spatial_bandwidth,
                                     kernel=self.kernel_)


        local_data_map = self.model_map.copy()
        n_total = len(site_id_arr)

        # ▶ 优化⑤：多进程并行训练局部模型
        def iter_task_args():
            for time_val,id_ in zip(times_arr,site_id_arr):

                neighbours_array, = np.where(neighbours[id_, :])

                con = np.logical_and(
                        np.abs(self._st_times_arr - time_val) < self.temporal_bandwidth,
                        np.isin(self._st_site_ids_arr, neighbours_array)
                )

                local_ds = global_data_set[con].sort_values(['time', 'site_id'])
                local_X = local_ds.drop(columns=['y', 'time', 'site_id'])
                local_y = local_ds['y']
                time_slices = np.sort(np.unique(self._st_times_arr[con]))

                yield (
                    time_val, id_,
                    weighted[id_, :],
                    local_X, local_y,
                    self._st_times_arr, self._st_site_ids_arr,
                    time_slices, neighbours_array,
                    self.bootstrap, self.train_weighted, self.resampled,
                    self.kernel_, self.pow,  self.temporal_bandwidth
                )

                # 显式释放当前循环里的临时对象
                del local_ds, local_X, local_y, time_slices, neighbours_array, con

        worker = partial(_construct_weight_worker,
                         local_model = self.local_model)

        with Pool(processes=self.workers) as pool:
            for idx, local_train_data in tqdm(
                    pool.imap_unordered(
                        worker,
                        iter_task_args(),
                        chunksize=1
                    ),
                    total=n_total,
                    desc='matching data and weight'
            ):
                local_data_map[idx] = local_train_data

        worker = partial(model_fit_importance,
                         model = self.local_model)
        self.importance_map = []
        with Pool(processes=self.workers) as pool:
            for idx, local_importance in tqdm(
                    pool.imap_unordered(
                        worker,
                        local_data_map.items(),
                        chunksize=1
                    ),
                    total=n_total,
                    desc='fitting local models'
            ):
                local_importance = pd.DataFrame([local_importance],columns=X_train.columns)
                local_importance['time'] = idx[0]
                local_importance['site'] = idx[1]
                self.importance_map.append(local_importance)
        self.importance_map = pd.concat(self.importance_map)
        return self.importance_map