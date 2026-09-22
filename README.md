# GTW-Models
Here is a framework named Geographically and temporally weighted models, enabling using customized local models such as random forest, extra-trees to overcome spatial non-stationarity and temporal variation. Those two mentioned problems are common while spatial modeling with aspatial models.

## Examples
```bash
    params = {
        'global_model': ExtraTreesRegressor(n_estimators=30,max_depth=30,random_state=42),
        'local_model': ExtraTreesRegressor(n_estimators=30,max_depth=30,random_state=42),
        'spatial_bandwidth':spatial_bandwidth,
        'kernel_': 'NN',
        'train_weighted': True,
        'test_weighted':False,
        'temporal_bandwidth': temporal_bandwidth,
        'local_weight': 0.5,
        'nworkers': -1
    }
```
