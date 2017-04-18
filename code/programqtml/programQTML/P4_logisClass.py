
#Fitting Multiple Linear Regression to the Training set


from sklearn.linear_model import LogisticRegression

def P4_modelfiting(X_train, y_train, X_test):    
    clf = LogisticRegression()
    clf.fit(X_train,y_train)
    
    #Predict
    y_pred = clf.predict(X_test)
    return y_pred

