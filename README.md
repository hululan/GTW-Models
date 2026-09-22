# GTW-Models
Here is a framework named Geographically and temporally weighted models, enabling using customized local models such as random forest, extra-trees to overcome spatial non-stationarity and temporal variation. Those two mentioned problems are common while spatial modeling with aspatial models.

## Examples
```bash
    params = {
        'global_model': ExtraTreesRegressor(n_estimators=30,max_depth=30,random_state=42),
        'local_model': ExtraTreesRegressor(n_estimators=30,max_depth=30,random_state=42),
        'spatial_bandwidth':31,
        'temporal_bandwidth': 3,
        'kernel_': 'NN',
        'train_weighted': True,
        'test_weighted':False,
        'local_weight': 0.5,
        'nworkers': -1
    }
```

Training model:
```bash
            model = GTWRF(
                **params
            )
            y_pred = model.fit_predict(
                X_train=X_train,
                y_train=y_train,
                site_ids_train=site_train,
                coords_train=coords_train,
                times_train=time_train,
                X_test=X_test,
                coords_test=coords_test,
                times_test=time_test)
```
