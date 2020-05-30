import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Users/gutanweer/Desktop/RNN/Recurrent_Neural_Networks/Google_Stock_Price_Train.csv")
train=df['Open'].values
train=train.reshape(-1,1)

#feature scaling
import sklearn
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
train_sc=sc.fit_transform(train)

#creating data structure for 60 timesteps
X_train=[]
y_train=[]
for i in range(60,1258):
    X_train.append(train_sc[i-60:i,0])
    y_train.append(train_sc[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)
X_train=np.reshape(X_train,(1198,60,1))

##model building
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

keras=Sequential()
keras.add(LSTM(units=60,return_sequences=True,input_shape=(60,1)))
keras.add(Dropout(0.2))

keras.add(LSTM(units=60,return_sequences=True))
keras.add(Dropout(0.2))

keras.add(LSTM(units=60,return_sequences=True))
keras.add(Dropout(0.2))

keras.add(LSTM(units=60,return_sequences=False))
keras.add(Dropout(0.2))

# Adding the output layer
keras.add(Dense(units = 1,))

# Compiling the RNN
keras.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
keras.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2020
dataset_test = pd.read_csv('C:/Users/gutanweer/Desktop/RNN/Recurrent_Neural_Networks/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2020
dataset_total = pd.concat((df['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = keras.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
a=len(dataset_total)
a
