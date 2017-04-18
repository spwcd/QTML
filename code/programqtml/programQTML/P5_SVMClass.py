#Fitting Support Vector Machine classification to the Training set


from sklearn import svm

def P5_modelfiting(X_train, y_train, X_test):    
    clf = svm.SVC()
    clf.fit(X_train,y_train)
    
    #Predict
    y_pred = clf.predict(X_test)
    return y_pred

