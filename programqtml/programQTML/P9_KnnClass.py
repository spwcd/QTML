#Fitting K-NN classification to the Training set

from sklearn.neighbors import KNeighborsClassifier

def P9_modelfiting(X_train, y_train, X_test): 
    KnnClf = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    KnnClf.fit(X_train, y_train)
    
    #Predict
    y_pred = KnnClf.predict(X_test)
    
    return y_pred
