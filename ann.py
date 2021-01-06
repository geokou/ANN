
# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Country column
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)

# Male/Female
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X = X[:, 1:]
 




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part 2 make ANN
#importing The Keras Libraries

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


#initialising the ANN
classifier = Sequential()

#adding the input layer and first hidden layer
classifier.add(Dense(6, activation = 'relu', input_dim = 11))
classifier.add(Dropout(rate = 0.1))
#adding second  hidden layer
classifier.add(Dense(6, activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

#adding the output layer
classifier.add(Dense(1, activation = 'sigmoid')) 

#compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

#Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs= 100)

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#--------new prediction
#France
#credit score 600
#Male
#age 40
#tenures 3
#balance 6000
#number of products 2
#credit card yes
#active member yes
#salary 50000
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pridection = (new_prediction >0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#K4 Evaluation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, activation = 'relu', input_dim = 11))
    classifier.add(Dense(6, activation = 'relu'))
    classifier.add(Dense(1, activation = 'sigmoid')) 
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
    return classifier
classifier   =  KerasClassifier(build_fn = build_classifier,batch_size = 10, epochs= 100 )
accurasies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accurasies.mean()
variance = accurasies.std()

#dropout overfitting if needed

#Tunning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, activation = 'relu', input_dim = 11))
    classifier.add(Dense(6, activation = 'relu'))
    classifier.add(Dense(1, activation = 'sigmoid')) 
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'] )
    return classifier
classifier   =  KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[25,32],
              'epochs':[100, 500],
              'optimizer':['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_



