import sys
sys.path.append('/Users/junenyjune/anaconda/lib/python3.5/site-packages')

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

import statsmodels.formula.api as sm
import talib as tl
from sklearn.preprocessing import StandardScaler

from tranform import *

# Importing the dataset
dataset = pd.read_csv('ADVANC.csv', parse_dates = True, index_col = 0)
dataset = dataset[['Open', 'High', 'Low', 'Close', 'Volume']]
dataset.dropna(inplace = True)

dataset['PCT_change'] = (dataset['Close'] - dataset['Close'].shift(1))/dataset['Close'].shift(1)*100.0

dataset['Open-1'] = dataset['Open'].shift(1)
dataset['High-1'] = dataset['High'].shift(1)
dataset['Low-1'] = dataset['Low'].shift(1)
dataset['Close-1'] = dataset['Close'].shift(1)
dataset['Volume-1'] = dataset['Volume'].shift(1)
dataset['PCT_change-1'] = dataset['PCT_change'].shift(1)
 
dataset.dropna(inplace=True)   
X = np.array(dataset[['Open-1','High-1','Low-1','Close-1','Volume-1']])
#feaScaling(X)
#sc =StandardScaler()
#Xnew = sc.fit_transform(X)

dataset['Label'] = dataset['PCT_change'].apply(labelUD)
y = np.array(dataset['Label'])



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import losses

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer

# output_dim = number of node in that hidden layer, init = format of initial weight, input_dim = number of input nodes
classifier.add(Dense(units=3, activation="relu", input_dim=5, kernel_initializer="uniform"))

# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=3, kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Fitting classifier to the Training set
# Create your classifier here


# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
#convert prob to 0 or 1


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
