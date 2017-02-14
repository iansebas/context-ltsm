import os, csv, argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.io import loadmat
from keras.layers.core import Dense, Activation, Dropout, Merge, Reshape, Flatten, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn import metrics
from numpy import arange, sin, pi, random
import pydot
from keras.utils.visualize_util import plot as kplt

np.random.seed(1234)

# Global hyper-parameters
num_states = 4
sequence_length = 20
memory_length = 16
represent_length = 8
epochs = 16
batch_size = 16
vroi = 100000
detect_thresh = 99
view_agent = 12

def load_sim(datapath):
    abs_path = os.getcwd()
    rel_path = datapath
    path = os.path.join(abs_path, rel_path)
    filename = open(path,'r+');
    raw_data = loadmat(filename,struct_as_record=False,mat_dtype=True)
    raw_data = np.array(raw_data.get('A'))
    print('Raw data size is...')
    print(raw_data.shape)

    return raw_data


def Series_Encoder():
    model = Sequential()

    model.add(LSTM(
            input_length=memory_length,
            input_dim=num_states,
            output_dim=represent_length,
            return_sequences=True))
    model.add(Dropout(0.2))

    return model

def Series_Decoder():
    model = Sequential()
    layers = {'input': represent_length, 'output': num_states}

    model.add(LSTM(
            input_length=memory_length,
            input_dim=layers['input'],
            output_dim=layers['output'],
            return_sequences=False))
    model.add(Dropout(0.2))

    return model


def Contextor():

    left = Sequential()
    left.add(LSTM(
            input_length=sequence_length,
            input_dim=num_states,
            output_dim=represent_length,
            return_sequences=True))
    left.add(Dropout(0.2))
    #left.add(TimeDistributedDense(represent_length))

    right = Sequential()
    right.add(LSTM(
            input_length=memory_length,
            input_dim=num_states,
            output_dim=represent_length,
            return_sequences=True))
    #right.add(TimeDistributedDense(represent_length))


    model = Sequential()
    model.add(Merge([left, right],mode='concat', concat_axis=1))

    model.add(LSTM(
            input_length=sequence_length + memory_length,
            input_dim=represent_length,
            output_dim=num_states,
            return_sequences=False))
    model.add(Dropout(0.2))

    return model



def build_models():
    net1 = Sequential()
    net1.add(Series_Encoder())
    net1.add(Series_Decoder())
    net2 = Contextor()

    start = time.time()
    net1.compile(loss="mse", optimizer="rmsprop")
    print "NN_1 Compilation Time : ", time.time() - start
    start = time.time()
    net2.compile(loss="mse", optimizer="rmsprop")
    print "NN_2 Compilation Time : ", time.time() - start
    return net1, net2


def plot_results(mse_matrix, anomaly_matrix, predictions, Universe,k,n):
    try:
        plt.figure(1)
        plt.plot(Universe[view_agent,:k,0], Universe[view_agent,:k,1], 'ro', predictions[view_agent,:k,0],predictions[view_agent,:k,1], 'bs')
        #plt.axis([0, 6, 0, 20])
        plt.savefig('figs/'+'map_'+'A'+str(view_agent)+'_net_type'+str(n)+'_time'+str(k))
        plt.figure(2)
        plt.plot(range(anomaly_matrix.shape[1])[:k], anomaly_matrix[view_agent,:k], 'ro')
        #plt.axis([0, 6, 0, 20])
        plt.savefig('figs/'+'detect_'+'A'+str(view_agent)+'_net_type'+str(n)+'_time'+str(k))
    except Exception as e:
        print("plotting exception")
        print str(e)

def run_networks(dataset):
    global_start_time = time.time()

    raw_data  = load_sim(dataset)
    Universe = raw_data[:,:,1:]


    print '\nData Loaded. Compiling...\n'

    net_1, net_2 = build_models()

    series_memory = np.zeros((Universe.shape[0],memory_length,num_states))
    state_memory = np.zeros((sequence_length,num_states))
    vector_memory = np.zeros((sequence_length,represent_length))
    k = Universe.shape[1]
    n = Universe.shape[0]

    mse_matrix_1 = np.zeros((n,k))
    mse_matrix_2 = np.zeros((n,k))
    anon_matrix_1 = np.zeros((n,k))
    anon_matrix_2 = np.zeros((n,k))
    predictions_1 = np.zeros((n,k,num_states))
    predictions_2 = np.zeros((n,k,num_states))


    try:
        for t in range(k)[1:-2]:
            print("Running Sim and Training Neural Networks Online...")
            observation = Universe[:,t,:].reshape((-1,num_states))
            net_1.fit(
                    series_memory, observation,
                    batch_size=batch_size, nb_epoch=epochs, validation_split=0.0625)
            predicted_1 = net_1.predict(series_memory,batch_size=16)

            net_2.fit(
                    [np.tile(state_memory,(series_memory.shape[0],1,1)),series_memory], observation,
                    batch_size=batch_size, nb_epoch=epochs, validation_split=0.0625)
            predicted_1 = net_1.predict(series_memory,batch_size=batch_size)
            predicted_2 = net_2.predict([np.tile(state_memory,(series_memory.shape[0],1,1)),series_memory],batch_size=batch_size)
            
            print("Anomaly Detection...")
            mse_1 = np.zeros((observation.shape[0]))
            mse_2 = np.zeros((observation.shape[0]))
            for elem in range(observation.shape[0]):
                mse_1[elem] = metrics.mean_squared_error(observation[elem,:],predicted_1[elem,:])
                mse_2[elem] = metrics.mean_squared_error(observation[elem,:],predicted_2[elem,:])

            anomaly_1 = mse_1 > np.percentile(mse_matrix_1[:,:t], detect_thresh)
            anomaly_2 = mse_2 > np.percentile(mse_matrix_2[:,:t], detect_thresh)


            mse_matrix_1[:,t] = mse_1
            mse_matrix_2[:,t] = mse_2
            anon_matrix_1[:,t] = anomaly_1
            anon_matrix_2[:,t] = anomaly_2
            predictions_1[:,t,:] = predicted_1
            predictions_2[:,t,:] = predicted_2

          
            print("Storing in memory...")
            if t < memory_length:
                series_memory[:,-t:,:] = Universe[:,0:t,:]
            else:
                series_memory = Universe[:,0+t:memory_length+t,:]

            states_t = raw_data[:,t,:].reshape((raw_data.shape[0],-1))
            states_t = states_t[states_t[:, 0].argsort()]
            states_t = states_t[np.ix_(states_t[:,0] < vroi, range(num_states+1)[1:])]
            states_t = states_t.reshape((-1,num_states))
            state_memory[:,-len(np.atleast_1d(states_t)):] = states_t

            logs = os.path.join(os.getcwd(),'logs/net_1_t'+str(k))
            if t % 500 == 0:
                print("Saving weights...")
                net_1.save_weights(logs)
                plot_results(mse_matrix_1,anon_matrix_1,predictions_1, Universe,t,1)
                plot_results(mse_matrix_2,anon_matrix_2,predictions_2, Universe,t,2)

    except KeyboardInterrupt:
        print("prediction exception")
        print 'Training duration (s) : ', time.time() - global_start_time
        return net_1, 0

    net_1.save_weights(logs)
    outfile = dataset + "_results_1"

    np.savez(outfile, mse_matrix_1=mse_matrix_1,anon_matrix_1=anon_matrix_1,predictions_1=predictions_1,Universe = Universe)
    plot_results(mse_matrix_1,anon_matrix_1,predictions_1, Universe,k,1)
    plot_results(mse_matrix_2,anon_matrix_2,predictions_2, Universe,k,2)

    print 'Training duration (s) : ', time.time() - global_start_time

    return net_1

def parse_args():
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('--dataset', dest='dataset', help='Choose dataset path', default='data/sim/m1.mat', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    run_networks(args.dataset)
