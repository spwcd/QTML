from sklearn.ensemble import RandomForestRegressor

def P6_modelfiting(X_train, y_train, X_test):    
    randomF_regr = RandomForestRegressor(n_estimators = 300, random_state = 0)
    randomF_regr.fit(X_train,y_train)
    
    #Predict
    y_pred = randomF_regr.predict(X_test)
    print(y_pred)
    return y_pred
