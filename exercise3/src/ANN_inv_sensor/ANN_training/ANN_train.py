#!/usr/bin/env python
# Code to train and optimise the neural network
# Optimisation will be carried out via Dropout and early stopping point
# Author: Charlie Street, Matthew Broadway

# try
# sudo apt-get install libatlas-base-dev
# then following the scipy docs to build numpy
# python setup.py build_ext --inplace to build such that it can be called from the directory
# https://docs.scipy.org/doc/numpy-1.10.4/user/install.html
# https://hunseblog.wordpress.com/2014/09/15/installing-numpy-and-openblas/

import sys
import numpy as np  # used for mathematical calculations
import matplotlib  # used for plotting results from tests
import csv  # used for reading in training data


# parameters to tune through various optimisation methods
NUM_INPUT_NEURONS = 6
NUM_HIDDEN_NEURONS = 70
DROPOUT_RATE = 0.8
RUN_DROPOUT = False
K_FOLDS_VALUE = 10  # pretty standard for k fold!
K_FOLDS_REPEATS = 1  # would ideally like more repeats but time is a serious issue!!!!
ETA = 0.1
EPOCHS = 3000

# file with all of the training data contained
TRAINING_FILE = 'training_data.csv'
IN_HIDDEN_WEIGHTS = 'in_hidden_overfit.csv'
HIDDEN_OUT_WEIGHTS = 'hidden_out_overfit.csv'
LOGGING = 'log_network.txt'
# 64 bit might be slightly faster
float_type = np.float64

# data from training set
class TrainingSet:
    __slots__ = ('X', 't')

    def __init__(self, X=None, t=None, copy=True):
        if X is not None and copy:
            self.X = np.matrix(X, dtype=float_type, copy=True)
        else:
            self.X = X

        if t is not None and copy:
            self.t = np.matrix(t, dtype=float_type, copy=True)
        else:
            self.t = t

    def copy(self):
        other = TrainingSet()
        other.X = np.matrix(self.X, dtype=float_type, copy=True)
        other.t = np.matrix(self.t, dtype=float_type, copy=True)
        return other

    def create_spoof(self, num_samples=10000):
        # create lists to start with
        Xl = [(0.45,0.5, 0.7, 0.3, 0.8, 0.3)] * num_samples
        tl = [1] * num_samples

        self.X = np.matrix(Xl, dtype=float_type)
        self.t = np.matrix(tl, dtype=float_type).transpose()


    def from_csv(self,filename):
        '''
        converts data from a csv file into a matrix form for calculation with the neural network
        '''
        with open(filename, 'rb') as train_file:
            reader = csv.reader(train_file)
            X_arr = []
            t_arr = []
            for row in reader:
                X_arr.append([row[0],row[1],row[2],row[3],row[4],row[5]])
                t_arr.append([row[6]])

        self.X = np.asmatrix(np.array(X_arr), dtype=float_type) # converting to numpy matrix as faster (and more space efficient)
        self.t = np.asmatrix(np.array(t_arr), dtype=float_type) # numpy is significantly faster according to tests I found on stack overflow


# weights for neural network
# initialise our weight vectors randomly
class Weights:
    __slots__ = ('in_hidden', 'hidden_out')

    def __init__(self, other_weights=None):
        if other_weights is None:
            self.randomize()
        else:
            self.in_hidden = np.matrix(other_weights.in_hidden, copy=True)
            self.hidden_out = np.matrix(other_weights.hidden_out, copy=True)

    def randomize(self):
        # randn draws from a normal distribution centered at 0. The arguments
        # are the dimensions
        self.in_hidden = np.matrix(
                np.random.randn(NUM_HIDDEN_NEURONS, NUM_INPUT_NEURONS),
                copy=True, dtype=float_type)
        self.hidden_out = np.matrix(
                np.random.randn(NUM_HIDDEN_NEURONS),
                copy=True, dtype=float_type)

    def copy(self):
        return Weights(self)

    def set_dummys(self,in_hidden_val, hidden_out_val):
        arr_1 = np.empty((NUM_HIDDEN_NEURONS, NUM_INPUT_NEURONS), dtype=float_type)
        arr_1.fill(in_hidden_val)
        arr_2 = np.empty(NUM_HIDDEN_NEURONS)
        arr_2.fill(hidden_out_val)
        self.in_hidden = np.asmatrix(arr_1, dtype=float_type)
        self.hidden_out = np.asmatrix(arr_2, dtype=float_type)

    def set_yourself(self,in_hidden,hidden_out):
        self.in_hidden = in_hidden
        self.hidden_out = hidden_out

def get_k_fold_set(training):
    '''take the full training set with target labels
       and split up into training and testing set
       the k value defines the size of the testing set with regards to the training set
       :return (training X, training t, testing X, testing t)
    '''
    global K_FOLDS_VALUE

    set_with_labels = np.concatenate((training.X,training.t),axis=1)  # should consist of rows of (x_i, t_i)
    np.random.shuffle(set_with_labels)  # shuffle them about a bit to have random folds
    size_of_fold = np.size(training.X[:,0])//K_FOLDS_VALUE  # get the size of the fold (require integer division)
    X_test = set_with_labels[0:size_of_fold,0:6]
    t_test = set_with_labels[0:size_of_fold,6]
    X_train = set_with_labels[size_of_fold:,0:6]
    t_train = set_with_labels[size_of_fold:,6]


    return X_train, t_train, X_test, t_test


def sigmoid(x):
    '''
    sigmoid/logistic function, output function for network
    '''
    return 1/(1 + np.exp(-x))


def cross_entropy(t_p, y_p):
    '''
    cross entropy loss function for neural network
    :param t_p: true value
    :param y_p: predicted value
    :return: cross entropy loss between true value and predicted value
    '''
    return -1 * (np.multiply(t_p, np.log(y_p)) + np.multiply((1-t_p), np.log(1-y_p)))


def forward_step(x_i, weights):
    '''
    carries out forward step of neural network
    :param x_i: input to neural network
    :param weights: weight object
    :return: the predicted output y_p from the neural network
    '''

    signals_to_hidden = weights.in_hidden * x_i.T  # do linear combination in one line for input to hidden
    out_of_hidden = np.tanh(signals_to_hidden)
    y_p = sigmoid((weights.hidden_out * out_of_hidden)[0, 0])  # linear combination from hidden to output layer; apply sigmoid
    return y_p

def new_epoch(epoch):
    #print('E ', end='')
    if (epoch+1) % 10 == 0:
        print 'E {}'.format(epoch+1)
    else:
        print 'E',
        sys.stdout.flush()

def back_propagate(train, weights):
    '''
    carries out back propagation on a neural network
    this is for a neural network used for binary classification
    :return: nothing. The return is the trained neural network
    '''

    for epoch in range(EPOCHS):  # have to iterate over the back propagation algorithm multiple times
        new_epoch(epoch)

        sample_order = np.arange(np.size(train.X[:,0]))
        np.random.shuffle(sample_order)  # randomise the order we get our training samples from
        for i in range(np.size(train.X[:, 0])):
            p = sample_order[i]
            x_p = np.asmatrix(train.X[p, :])  # training sample
            t_p = train.t[p]  # ground truth

            # forward step
            s_l1 = weights.in_hidden * x_p.T  # calculate signals going into hidden neurons
            a_l1 = np.tanh(s_l1)

            # carry out dropout, if we have opted to
            if RUN_DROPOUT:
                # select which neurons to set to 0 according to dropout rate selected
                # appropriate credit to be given to Shan He for demonstrating this method of implementing dropout
                selected_neurons = np.random.binomial(1, DROPOUT_RATE, (1,NUM_HIDDEN_NEURONS))
                a_l1 = np.multiply(a_l1, selected_neurons.T)  # actually make the change to temporarily remove some neurons
            s_l2 = (weights.hidden_out * a_l1)[0, 0]
            y_p = sigmoid(s_l2) # get the predicted value from the network

            # weight update for layer 2
            delta_l2 = y_p - t_p  # I have derived this for binary classification
            hidden_out_weight_update = -1 * ETA * a_l1 * delta_l2  # write this properly
            weights.hidden_out += hidden_out_weight_update.T  # apply weight update

            # weight update for layer 1 (hidden layer)
            one_minus_square = np.vectorize(lambda x: 1-x**2)
            one_minus_square_a_l1 = one_minus_square(a_l1)
            delta_l1 = delta_l2 * np.multiply(one_minus_square_a_l1.T , weights.hidden_out)
            in_hidden_weight_update = -1 * ETA * delta_l1.T * x_p
            weights.in_hidden += in_hidden_weight_update



def float_mat(m):
    ''' convert a matrix to have the correct representation for back propagation '''
    return np.asmatrix(np.ndarray.astype(m, dtype=float_type, copy=True, order='C'))

def tmp_mat(rows, cols):
    ''' create a matrix for holding temporary results '''
    return np.matrix(np.empty((rows, cols), dtype=float_type, order='C'),
            dtype=float_type, copy=False)

def print_dimensions(ma, mb, mc):
    print('ma = ', ma.shape)
    print('mb = ', mb.shape)
    print('mc = ', mc.shape)


# faster than the default of using a dict for object attributes
class NNLayer:
    __slots__ = ('s', 'a')
    def __str__(self):
        return 'NNLayer(s={}, a={})'.format(self.s, self.a)
    def __repr__(self):
        return self.__str__()


def back_propagate_fast(train, weights):
    '''
    carries out back propagation on a neural network
    this is for a neural network used for binary classification
    :return: nothing. The return is the trained neural network
    '''

    # allocate variables to hold temporary results
    num_samples = train.X.shape[0] # num rows
    xs = [np.matrix(train.X[p, :], copy=False) for p in range(num_samples)]

    samples = np.array([(xs[p], xs[p].T, train.t.item(p,0)) for p in range(num_samples)])

    l1 = NNLayer()
    l1.s = tmp_mat(weights.in_hidden.shape[0], 1)
    l1.a = l1.s # re-use l1.s since it isn't used after assignment

    l2 = NNLayer()
    l2.s = tmp_mat(weights.hidden_out.shape[0], l1.a.shape[1])
    l2.a = None

    hidden_out_weight_update = tmp_mat(*l1.a.shape)
    delta_l1 = tmp_mat(l1.a.shape[1], weights.hidden_out.shape[1]) # vector
    in_hidden_weight_update = tmp_mat(delta_l1.shape[1], train.X.shape[1])

    # transposes (makes shallow copy which shares data)
    hidden_out_weight_update_T = hidden_out_weight_update.T
    l1_a_T = l1.a.T
    delta_l1_T = delta_l1.T

    for epoch in range(EPOCHS):  # have to iterate over the back propagation algorithm multiple times
        new_epoch(epoch)

        np.random.shuffle(samples)  # randomise the order we get our training samples from

        for x_p, x_p_T, t_p in samples:
            # x_p is a row vector of a training sample
            # x_p_T is a column vector
            # t_p is ground truth

            # forward step
            np.dot(weights.in_hidden, x_p_T, out=l1.s) # calculate signals going into hidden neurons
            np.tanh(l1.s, out=l1.a)

            # carry out dropout, if we have opted to
            if RUN_DROPOUT:
                # select which neurons to set to 0 according to dropout rate selected
                # appropriate credit to be given to Shan He for demonstrating this method of implementing dropout
                selected_neurons = np.random.binomial(1, DROPOUT_RATE, (1, NUM_HIDDEN_NEURONS))
                np.multiply(l1.a, selected_neurons.T, out=l1.a)  # actually make the change to temporarily remove some neurons

            # l2.s is a 1x1 matrix
            np.dot(weights.hidden_out, l1.a, out=l2.s)
            y_p = sigmoid(l2.s.item(0)) # get the predicted value from the network

            # weight update for layer 2
            delta_l2 = y_p - t_p  # I have derived this for binary classification
            # first argument is a scalar, second is a vector
            np.multiply(-1 * ETA * delta_l2, l1.a, out=hidden_out_weight_update) # write this properly
            weights.hidden_out += hidden_out_weight_update_T  # apply weight update

            # weight update for layer 1 (hidden layer)
            # modify l1.a in-place. Element wise operation: 1-x^2
            #np.power(l1.a, 2, out=l1.a)
            np.multiply(l1.a, l1.a, out=l1.a) # faster than pow
            np.subtract(1, l1.a, out=l1.a)

            np.multiply(l1_a_T, weights.hidden_out, out=delta_l1)
            delta_l1 *= delta_l2

            np.multiply(delta_l1_T, x_p, out=in_hidden_weight_update)
            in_hidden_weight_update *= -1 * ETA
            weights.in_hidden += in_hidden_weight_update




def get_avg_kfold_err(training,weights):
    '''
    gets avg kfold error for a network trained with a generated subset of the full training data,
             with multiple repeats
    :return: avg kfold error
    '''
    total_sum = 0.0  # will store final result
    for _ in range(K_FOLDS_REPEATS):  # do multiple repeats to reduce validation bias
        local_sum = 0.0
        for _ in range(K_FOLDS_VALUE):  # do for each of the k folds
            fold_sum = 0.0
            X_train, t_train, X_test, t_test = get_k_fold_set(training)  # get the partition

            weights_copy = weights.copy()
            back_propagate_fast(TrainingSet(X_train,t_train),weights_copy)  # train the network with the training set partition
            check = weights_copy.in_hidden != weights.in_hidden
            check2 = weights_copy.hidden_out != weights.hidden_out
            print('checking matrices are different ' + str(check) + ' ' + str(check2))
            # get total error for this fold
            for i in range(np.size(X_test[:, 0])):
                fold_sum += cross_entropy(t_test[i], forward_step(X_test[i, :],weights_copy))
            local_sum += fold_sum  # add total fold error to local sum, this will be averaged over k folds later

        total_sum += (local_sum / float(K_FOLDS_VALUE))
    return total_sum / float(K_FOLDS_REPEATS)


def get_avg_training_err(train,weights):
    '''
    as part of using dropout as a regularisation method we first need to over-fit our training data greatly
    in order to ensure we over-fit in this case, we can use the training error to give us some measure
    :return: the average training error over all training samples for the current weight vectors
    '''

    total_sum = 0.0
    no_samples = np.size(train.X[:,0])
    for p in range(no_samples):
        total_sum += cross_entropy(train.t[p],forward_step(train.X[p,:],weights))  # loss for a given sample

    return total_sum / float(no_samples)  # get average of all losses for individual training samples

# backups
'''train = TrainingSet()
train.create_spoof()
w = Weights()

EPOCHS = 100
import time

# slow
print('timing back propagation')
s_tr = train.copy()
s_w = Weights(w)
start = time.time()
back_propagate(s_tr, s_w)
slow_time = time.time()-start
print('\ntook {} seconds for {} epochs'.format(slow_time, EPOCHS))


# fast
print('timing fast back propagation')
f_w = Weights(w)
start = time.time()
back_propagate_fast(train, f_w)
fast_time = time.time()-start
print('\ntook {} seconds for {} epochs'.format(fast_time, EPOCHS))

print
print('speedup = {}'.format(slow_time/fast_time))
print

if float_type == np.float32:
    print('weights.in_hidden\'s: {}'.format(np.allclose(s_w.in_hidden, f_w.in_hidden, rtol=1e-3)))
    print('weights.hidden_out\'s: {}'.format(np.allclose(s_w.hidden_out, f_w.hidden_out, rtol=1e-3)))
else:
    print('weights.in_hidden\'s: {}'.format(np.allclose(s_w.in_hidden, f_w.in_hidden)))
    print('weights.hidden_out\'s: {}'.format(np.allclose(s_w.hidden_out, f_w.hidden_out)))

#print('compare with originals')
#print('weights.hidden_out\'s: {}'.format(np.allclose(w.in_hidden, s_w.in_hidden)))
#print('weights.hidden_out\'s: {}'.format(np.allclose(w.hidden_out, s_w.hidden_out)))

#x_train,t_train,x_test,t_test = get_k_fold_set()

'''

def overfit_network():
    '''function will overfit the network'''
    training = TrainingSet()
    training.from_csv(TRAINING_FILE)  # read in data from training file
    weights = Weights()  # initialise some random weights
    original_weights = weights.copy()
    back_propagate_fast(training,weights)  # back propagate
    np.savetxt('in_hidden_overfit.csv', weights.in_hidden, delimiter=",")
    np.savetxt('hidden_out_overfit.csv', weights.hidden_out, delimiter=",")
    error = get_avg_training_err(training,weights)
    print('Average training error: ' + str(error))  # let me know the average error


def main(job_id,params):
    '''function for bayesian optimisation'''
    '''training = TrainingSet()
    training.from_csv(TRAINING_FILE)
    EPOCHS = params['EPOCHS']
    ETA = params['ETA']
    DROPOUT_RATE = params['DROPOUT_RATE']
    in_hidden = np.asmatrix(np.loadtxt(IN_HIDDEN_WEIGHTS,delimiter=","))
    hidden_out = np.asmatrix(np.loadtxt(HIDDEN_OUT_WEIGHTS,delimiter=","))
    weights = Weights()
    weights.set_yourself(in_hidden,hidden_out)

    k_fold_error = get_avg_kfold_err(training,weights)'''
    logger = open(LOGGING,'a')
    logger.write('Current Run: EPOCHS = ' + ' ETA = ' + 'DROPOUT RATE = ' + '. K Fold Error is ' + '\n')
    logger.close()
    #return k_fold_error

def run_manually():
    '''manually tunes dropout rate from overfitted network. This is being done in accordance with the research paper '''
    global DROPOUT_RATE, RUN_DROPOUT
    dropout_vals = [0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45]
    RUN_DROPOUT = True
    for i in dropout_vals:
        DROPOUT_RATE = i
        training = TrainingSet()
        training.from_csv(TRAINING_FILE)

        in_hidden = np.asmatrix(np.loadtxt(IN_HIDDEN_WEIGHTS, delimiter=","))
        hidden_out = np.asmatrix(np.loadtxt(HIDDEN_OUT_WEIGHTS, delimiter=","))
        weights = Weights()
        weights.set_yourself(in_hidden, hidden_out)

        k_fold_error = get_avg_kfold_err(training, weights)[0,0]
        logger = open(LOGGING,'a')
        logger.write('Current Run: EPOCHS = ' + str(EPOCHS) + ' ETA = ' + str(ETA) + ' DROPOUT RATE = ' + str(DROPOUT_RATE) + '. K Fold Error is ' + str(k_fold_error) + '\n')
        logger.close()


def create_best_network():
    '''function will run the back propagation algorithm for our best set of parameters and then save the weights'''
    global ETA, DROPOUT_RATE, EPOCHS, RUN_DROPOUT, NUM_HIDDEN_NEURONS
    NUM_HIDDEN_NEURONS = 50
    ETA = 0.1  # my optimal parameters
    DROPOUT_RATE = 0.7
    EPOCHS = 2000
    RUN_DROPOUT = True
    training = TrainingSet()  # load in the training data
    training.from_csv(TRAINING_FILE)

    in_hidden = np.asmatrix(np.loadtxt('best_network/overfitted_2000_50/in_hidden_overfit.csv', delimiter=","))  # the best overfitted weights
    hidden_out = np.asmatrix(np.loadtxt('best_network/overfitted_2000_50/hidden_out_overfit.csv', delimiter=","))
    weights = Weights()
    weights.set_yourself(in_hidden,hidden_out)


    back_propagate_fast(training,weights)  # run the training algorithm

    # now multiply the output layer weights by the dropout rate in accordance with the research paper
    optimised_in_hidden = weights.in_hidden * DROPOUT_RATE
    optimised_hidden_out = weights.hidden_out

    np.savetxt('best_network/in_hidden_optimised.csv', optimised_in_hidden, delimiter=",")
    np.savetxt('best_network/hidden_out_optimised.csv', optimised_hidden_out, delimiter=",")
    print('Optimal weight matrices found and saved\n')

#overfit_network()
#run_manually()
#create_best_network()
K_FOLDS_VALUE = 10
EPOCHS = 12000
NUM_HIDDEN_NEURONS = 15
ETA = 0.1
RUN_DROPOUT = False

training = TrainingSet()
training.from_csv(TRAINING_FILE)
weights = Weights()
back_propagate_fast(training,weights)
np.savetxt('network_attempts/12000_15epochs_in_hidden.csv', weights.in_hidden, delimiter=",")
np.savetxt('network_attempts/12000_15epochs_hidden_out.csv', weights.hidden_out, delimiter=",")
#result = get_avg_kfold_err(training,weights)
#check1 = weights.in_hidden == weights_copy.in_hidden
#check2 = weights.hidden_out == weights_copy.hidden_out

#print(str(check1))
#print(str(check2))
#print('K fold error is: ' + str(result))
