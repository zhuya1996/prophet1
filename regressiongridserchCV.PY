xgb_model = xgb.XGBRegressor(nthread=10)
clf = pipeline.Pipeline([('xgb', xgb_model)])
param_grid = {'xgb__max_depth': [1,2,3],    
              'xgb__learning_rate': [0.2,0.3,0.4],  
              'xgb__n_estimators': [40,45,50]   
             }  

model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, verbose=10, n_jobs=1,
                         iid=True, refit=True, cv=10)

model.fit(X, Y_log)
