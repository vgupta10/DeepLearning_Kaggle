import pandas as pd
import csv
import numpy as np
import tensorflow
import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience = 2)


'''trainingTest is dataset that contains all the training dataset and test dataset vertically stacked up
 essentially reading in both datasets and we will split them later.
does not include the response variable for the training dataset'''
df = pd.read_csv('trainingTest.csv') 


#extracts the response variable from just the training dataset, test response not available
targetDF = pd.read_csv('training.csv')
target = np.array(targetDF.ix[:,"loss"])

#data exploration
df = df.drop('Unnamed: 0', axis = 1)
df2 = df.drop('id', axis=1)


column_names_for_onehot = df2.columns[0:116] # all categorical columns
df2 = pd.get_dummies(df2, columns=column_names_for_onehot)


holdOutDF = df2.tail(125546) # validation
df2 = df2.head(188318) #training

#converting predictors to numpy matrix for input into keras model
predictors  = df2.as_matrix()
n_cols = predictors.shape[1] #number of columns/inputs into the network
target.shape[0]


#MODEL1
#starting to create model
#starting with simple 1 hidden layer network, first hidden layer has 50 nodes
# Set up the model: model
model = Sequential() #making a sequential NN
# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
# Add the output layer
model.add(Dense(1))
#compiling the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae'])
#ensuring we stop after n epochs, based on validation error
model.fit(predictors, target,validation_split = 0.2,epochs = 15)
predictions = model.predict(predictors)
mean_absolute_error(target, predictions) #checking how it does on training data
#epoch = 15 is the best



#experimenting with second model, add another hidden layer, widening network
# Set up the model: model
model2 = Sequential() #making a sequential NN
# Add the first layer
model2.add(Dense(150, activation='relu', input_shape=(n_cols,)))
model2.add(Dense(100, activation='relu'))
# Add the output layer
model2.add(Dense(1))
#compiling the model
model2.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae'])
#ensuring we stop after n epochs, based on validation error
model2.fit(predictors, target,validation_split = 0.2,epochs = 10)
predictions = model2.predict(predictors)
mean_absolute_error(target, predictions) #checking how it does on training data
#epoch = 10 is the best


#experimenting with third model, #adding two more hidden layers, for a total of 4
# Set up the model: model
model3 = Sequential() #making a sequential NN
# Add the first layer
model3.add(Dense(350, activation='relu', input_shape=(n_cols,)))
model3.add(Dense(300, activation='relu'))
model3.add(Dense(200, activation='relu'))
model3.add(Dense(100, activation='relu'))
model3.add(Dense(100, activation='relu'))
model3.add(Dense(100, activation='relu'))
model3.add(Dense(100, activation='relu'))
# Add the output layer
model3.add(Dense(1))
#compiling the model
model3.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae'])
#ensuring we stop after n epochs, based on validation error
model3.fit(predictors, target,validation_split = 0.2,epochs = 7)
predictions = model3.predict(predictors)
mean_absolute_error(target, predictions) #checking how it does on training data
#epoch = 11 is the best


#fourth model, 8 hidden layers in total, less widening the layers
# Set up the model: model
model4 = Sequential() #making a sequential NN
# Add the first layer
model4.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model4.add(Dense(80, activation='relu'))
model4.add(Dense(65, activation='relu'))
model4.add(Dense(50, activation='relu'))
model4.add(Dense(30, activation='relu'))
model4.add(Dense(20, activation='relu'))
model4.add(Dense(10, activation='relu'))
model4.add(Dense(5, activation='relu'))
# Add the output layer
model4.add(Dense(1))
#compiling the model
model4.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae'])
#ensuring we stop after n epochs, based on validation error
model4.fit(predictors, target,epochs =3)
predictions = model4.predict(predictors)
#epoch = 5 is the best #1168 error rate
#model4 is the best so far



testPredictors = holdOutDF.as_matrix()
predictions = model4.predict(testPredictors) #predictions for the test dayaset
predictions = predictions.tolist()
predictions = sum(predictions, [])
predictions.insert(0,predictions[0])

with open("predictions.csv", "w") as output: # writing predictions to csv file to submit
    writer = csv.writer(output, lineterminator='\n')
    for val in predictions:
        writer.writerow([val])    
