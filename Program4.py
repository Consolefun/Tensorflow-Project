from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#read in training data
dataset = pd.read_csv('./heart.csv')
seed = 5

Category_list = ["sex", "fbs", "restecg", "exang", "cp", "slope", "ca", "thal", "target"]
Truevalue_list = ["age", "trestbps", "chol", "thalach", "oldpeak"]

#one-hot encode target column

target = to_categorical(dataset.target)

thal = to_categorical(dataset.thal)

ca = to_categorical(dataset.ca)

slope = to_categorical(dataset.slope)

cp = to_categorical(dataset.cp)

exang = to_categorical(dataset.exang)

restecg = to_categorical(dataset.restecg)

fbs = to_categorical(dataset.fbs)

sex = to_categorical(dataset.sex)

one_hot_list = [sex, fbs, restecg, exang, cp, slope, ca, thal, target]

for category in Category_list:
    dataset = dataset.drop([category], axis=1)

#Normalization
for feature_name in Truevalue_list:
    max_value = dataset[feature_name].max()
    min_value = dataset[feature_name].min()
    dataset[feature_name] = (dataset[feature_name] - min_value) / (max_value - min_value)

dataset_numpy = dataset.values
for i in one_hot_list:
    dataset_numpy = np.append(dataset_numpy, i, 1)

#get number of columns in training data
n_cols_2 = dataset_numpy.shape[1]

# split into input and output variables
X = dataset_numpy[:,0:n_cols_2-2]
Y = dataset_numpy[:,n_cols_2-2:]

# split the data into training (70%) and testing (15%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.15, random_state=seed)

# create the model
model = Sequential()
model.add(Dense(n_cols_2-2, input_dim=n_cols_2-2, kernel_initializer='uniform', activation='sigmoid'))
model.add(Dense(10, kernel_initializer='uniform', activation='sigmoid'))
model.add(Dense(2, kernel_initializer='uniform', activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X_train, Y_train, validation_split= 0.15, epochs=200, batch_size=303, verbose=0)

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))    




    




