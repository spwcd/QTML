#Fitting Multiple Linear Regression to the Training set

from sklearn.linear_model import  LinearRegression

#import talib as tl


def P1_modelfiting(X_train, y_train, X_test):    
    clf = LinearRegression()
    #clf.OLS.fit(X_train,y_train)
    clf.fit(X_train,y_train)
    
    #Predict
    y_pred = clf.predict(X_test)
    print(y_pred)
    return y_pred

