
#Fitting Support Vector Regression to the Training set


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def P3_modelfiting(X_train, y_train, X_test):
    # Feature Scaling
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)
    
    # Fitting SVR to the dataset
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train, y_train)    

    # predict
    y_pred = regressor.predict(X_test)
    y_pred = sc_y.inverse_transform(y_pred)
    
    return y_pred
