import numpy
import math
import time
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#shape timeseries according to lookback parameter
def shape_dataset(dataset, look_back = 1):
	dataX, dataY = [], []

	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])

	return numpy.array(dataX), numpy.array(dataY)

def create_model(trainX, trainY,look_back = 1):
    no_epochs = 200
    batchsize = 20
    optimizer = Adam(lr=0.008, amsgrad=False)

    model = Sequential()

    layer1 = LSTM(6, input_shape=(1, look_back), return_sequences=True)
    model.add(layer1)

    layer2 = LSTM(6, input_shape=(1, look_back))
    model.add(layer2)

    layer3 = Dense(1, activation='sigmoid')
    model.add(layer3)

    #comlile and fit the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    t0 = time.time()
    model.fit(trainX, trainY, epochs=no_epochs, batch_size=batchsize, verbose=2, shuffle=True)
    t1 = time.time()

    print("Training time: %.2f sec." %(t1-t0))
    #save the model into a .h5 file
    model.save('lstm_model.h5')

    print(layer1.get_weights() ,file=open("layer1_weights.txt", "a"))
    print(layer2.get_weights(), file=open("layer2_weights.txt", "a"))
    print(layer3.get_weights(), file=open("layer3_weights.txt", "a"))

    return model

def training_phase():
    #fix seed for reproducibility
    numpy.random.seed(7)

    #read dataframe and fill the NAN values with 0
    dataframe = pd.read_csv("inputs.csv", sep='\t', encoding='utf-8')
    dataframe = dataframe.fillna(0)
    dataframe = dataframe.drop(dataframe.columns[0], axis = 1)

    #dataset to numpy array
    dataset = dataframe.values

    #reshape dataset from multiple timeseries to a single one
    dataset = numpy.reshape(dataset, (dataset.shape[0]*dataset.shape[1], 1))

    #smooth the timeseries by setting the zero values to a mean of its previous
    # and next value
    for i in range(len(dataset)):
    	if dataset[i] == 0 :
    		if i > 0 and i < len(dataset) -1 :
    			dataset[i] = int((dataset[i-1] + dataset[i+1]) / 2)

    #use variations instead of the real values
    for i in range(len(dataset)-1):
    	dataset[i] = abs(dataset[i] - dataset[i+1])

    dataset = dataset.astype('float32')

    #scale the values in range of 0 to 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    #split the dataset into train and test set
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size

    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    #shape the dataset according to the needs of LSTM Neural Network
    look_back = 5
    trainX, trainY = shape_dataset(train, look_back)
    testX, testY = shape_dataset(test, look_back)

    #reshape the dataset to meet the needs of LSTM NN input [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    #create LSTM Network
    model = create_model(trainX, trainY, look_back)

    print("Model Summary")
    print(model.summary())

    #make predictions to test the model

    trainPredict = model.predict(trainX)

    testPredict = model.predict(testX)

    #invert predictions and the target values
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    #round the values to get the citation value
    for i in range(len(trainPredict)):
    	trainPredict[i][0] = round(trainPredict[i][0])

    for i in range(len(testPredict)):
    	testPredict[i][0] = round(testPredict[i][0])

    #calculate Root Mean Squared Error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))

    print('Train Score: %.2f RMSE' % (trainScore))

    testScore = math.sqrt(mean_squared_error(testY[0],  testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

    # plot initial line and the predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()


training_phase()
