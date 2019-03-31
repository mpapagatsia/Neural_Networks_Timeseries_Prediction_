import numpy
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import time
import glob
import errno

# Reshape dataset to meet the needs of the model
def reshape_timeseries(timeseries):
    look_back = 5

    dataX = []
    i = len(timeseries)-1
    while True:
        a = timeseries[(i-look_back+1):i+1]
        if len(a) < look_back:
            break
        i -= 1
        dataX.append(a)
    dataX = numpy.array(dataX)
    dataX = numpy.flip(dataX)
    timeseries = numpy.reshape(dataX, (dataX.shape[0], 1, look_back))
    return timeseries

model = load_model('lstm_model.h5')
timeseries = []

path = '/home/katerina/Desktop/NeuralNetworks/project/neural_networks/timeseries*.txt'
years = 5

# Read all files with timeseries and store them in a list "timeseries"
# Example format of time series given : 1,3,4,5,7,8
files = glob.glob(path)
for name in files:
    ts = []
    try:
        with open(name) as f:
            ts = f.read()
            ts = ts.rstrip()
            ts = ts.split(',')
            print len(ts)
            tmp = []
            for i in range(years):
                tmp.append(int(ts[len(ts)-1]))
                del ts[len(ts)-1]
            ts = [int(x) for x in ts]
            tmp.reverse()
            timeseries.append(ts)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

# Convert each timeseries to a numpy array with float data
for i in range(len(timeseries)):
    timeseries[i] = numpy.array(timeseries[i])
    timeseries[i] = timeseries[i].astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))

predictions = []

# For every timeseries we compute the prediction for the next 5 years
for i in range(len(timeseries)):
    for j in range(years):
        reshaped_data = numpy.reshape(timeseries[i], (timeseries[i].shape[0], 1))
        reshaped_data = scaler.fit_transform(reshaped_data)
        reshaped_data = reshape_timeseries(reshaped_data)

        start = time.time()
        output_data = model.predict(reshaped_data)
        end = time.time()
        print "Time required for prediction: ", end-start

        output_data = scaler.inverse_transform(output_data)
        output_data = [numpy.round(a,0) for a in output_data]

        timeseries[i] = numpy.append(timeseries[i], output_data[len(output_data)-1])

        print "Estimated value: ", output_data[len(output_data)-1]
        print "\n"
