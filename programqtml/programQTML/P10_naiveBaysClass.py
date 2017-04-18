#Fitting Naive Bayes classification to the Training set

from sklearn.naive_bayes import GaussianNB

def P10_modelfiting(X_train, y_train, X_test): 
    naiveClf = GaussianNB()
    naiveClf.fit(X_train, y_train)
    
    #Predict
    y_pred = naiveClf.predict(X_test)
    
    return y_pred
    