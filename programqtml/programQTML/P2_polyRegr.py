#Fitting Multiple polynomial Regression to the Training set

from sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from dataPrep import degree

def P2_modelfiting(X_train, y_train, X_test): 
    degr = degree()
    poly_reg = PolynomialFeatures(degree = degr)
    X_poly = poly_reg.fit_transform(X_train)
    poly_reg.fit(X_poly, y_train)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y_train)
    
    # predict
    y_pred = lin_reg.predict(poly_reg.fit_transform(X_test))
    
    return y_pred
    
