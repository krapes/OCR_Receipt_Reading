# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:23:20 2017

@author: KRapes
"""

import tensorflow as tf
import dataset_lib
import numpy as np
from sklearn.gaussian_process import GaussianProcess
import os
import matplotlib.pyplot as plt
import random
from time import time

colors = plt.rcParams['axes.color_cycle']


CURRENT_DIR = os.path.abspath(os.path.dirname(__name__))
SAVE_MODEL_PATH = "./EMNIST_classification"
LOCAL_DATASET = False

def file_names(folder):
    files = []
    for file in os.listdir(folder):
        if file.startswith("labeled_images_"):
            files.append(os.path.join(folder, file.split('.')[0]))
    return files


def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    image_height = image_shape[0]
    image_width = image_shape[1]
    color_channels = image_shape[2]
    imagetensor = tf.placeholder(tf.float32, shape=[None, image_height, image_width, color_channels],name='x')
    return imagetensor


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    labelstensor = tf.placeholder(tf.float32, shape=[None,n_classes], name='y')
    return labelstensor


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    keeptensor = tf.placeholder(tf.float32,name='keep_prob')
    return keeptensor
    
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function
    #print("x_tensor: {} \nconv_num_outputs: {} \nconv_ksize: {} \nconv_strides: {}" .format(x_tensor, conv_num_outputs, conv_ksize, conv_strides))
    #print("pool_ksize: {} \npool_strides: {}" .format(pool_ksize, pool_strides))
   
    # Naming conventions
    filter_size_height = conv_ksize[0]
    filter_size_width = conv_ksize[1]
    color_channels = int(x_tensor.shape[3])
    k_output = conv_num_outputs
    stride = [1,conv_strides[0],conv_strides[1],1]
    
    #creating weight
    weight = tf.Variable(tf.truncated_normal(
    [filter_size_height, filter_size_width, color_channels, k_output]))
    
    #creating bias
    bias = tf.Variable(tf.zeros(k_output))
    
    # Apply Convolution
    conv_layer = tf.nn.conv2d(x_tensor, weight, strides=stride, padding='VALID')
    # Add bias
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    # Apply activation function
    conv_layer = tf.nn.relu(conv_layer)
    
    normalized = tf.layers.batch_normalization (conv_layer)
    
    # max pooling
    ksize = [1, pool_ksize[0], pool_ksize[1], 1]
    stride = [1, pool_strides[0], pool_strides[1], 1]
    padding = 'VALID'
    pool_layer = tf.nn.max_pool(normalized, ksize, stride, padding)
    
    return pool_layer 
    
def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    import numpy as np
    #print(x_tensor)

    shape = x_tensor.get_shape().as_list()        # a list: [None, height, width, channels]
    dim = np.prod(shape[1:])            # dim = prod(height,width,channels) 
    flattened_tensor = tf.reshape(x_tensor, [-1, dim])           # -1 means "all"
    #print(flattened_tensor)
    return flattened_tensor

def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    x_tensor = tf.contrib.layers.fully_connected(inputs=x_tensor, num_outputs=num_outputs, activation_fn=tf.nn.relu)
    return x_tensor
    
def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    tensor = tf.contrib.layers.fully_connected(inputs=x_tensor, num_outputs=num_outputs, activation_fn=None)
    return tensor
    
def conv_net(x, keep_prob, nconv1, nconv2, nfullyconn, nfullyconn2):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    #layer_norm = tflearn.layers.normalization.batch_normalization (x,  name='BatchNormalization')
    layer_conv = conv2d_maxpool(x, nconv1, (2,2), (2,2), (2,2), (2,2))
    #layer_conv = tf.nn.dropout(layer_conv, keep_prob)
    layer_conv = tf.layers.batch_normalization (layer_conv,  name='BatchNormalization')
    #layer_conv = tf.nn.dropout(layer_conv, keep_prob)
    #print(layer_conv)
    layer_conv = conv2d_maxpool(x, nconv2, (5,5), (2,2), (2,2), (2,2))
    layer_conv = tf.layers.batch_normalization (layer_conv,  name='BatchNormalization2')
    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    layer_flat = flatten(layer_conv)
    #layer_flat = tflearn.layers.normalization.batch_normalization (layer_flat,  name='BatchNormalization')
    
    

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    #layer_fully_conn = fully_conn(x, nfullyconn)
    layer_fully_conn = fully_conn(layer_flat, nfullyconn)
    #print("Fully Connected Outputs: {}".format(layer_fully_conn.shape[1]))
    #layer_fully_conn = fully_conn(layer_fully_conn, nconv)
    layer_fully_conn = tf.layers.batch_normalization (layer_fully_conn,  name='BatchNormalization3')
    layer_flat = flatten(layer_fully_conn)
    layer_fully_conn = fully_conn(layer_flat, nfullyconn2)
    layer_fully_conn = tf.layers.batch_normalization (layer_fully_conn,  name='BatchNormalization4')
    layer_flat = flatten(layer_fully_conn)
    layer_fully_conn = tf.nn.dropout(layer_fully_conn, keep_prob)
    #layer_fully_conn = tf.nn.dropout(layer_fully_conn, keep_prob)
    
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    layer_final = output(layer_fully_conn, 46)
    
    
    # TODO: return output
    return layer_final



# Remove previous weights, bias, inputs, etc..
def build_network(nconv1, nconv2, nfullyconn, nfullyconn2):
    tf.reset_default_graph()

    # Inputs
    x = neural_net_image_input((28, 28, 1))
    y = neural_net_label_input(46)
    keep_prob = neural_net_keep_prob_input()

    # Model
    logits = conv_net(x, keep_prob, nconv1, nconv2, nfullyconn, nfullyconn2)

    # Name logits Tensor, so that is can be loaded from disk after training
    logits = tf.identity(logits, name='logits')

    # Loss and Optimizer
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    #cost = -tf.reduce_sum(logits*tf.log(tf.clip_by_value(y,1e-10,1.0)))
    cost = -tf.reduce_sum(y*tf.log(tf.nn.softmax(logits) + 1e-10))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    
    return x, y, keep_prob, cost, optimizer, accuracy
    

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch, x, y, keep_prob):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    session.run(optimizer, feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})
    pass

def print_stats(session, feature_batch, label_batch, cost, accuracy, x, y, keep_prob, valid_features, valid_labels):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    result_cost = session.run(cost, feed_dict={x:feature_batch, y:label_batch, keep_prob: 1.0})
    result_accuracy = session.run(accuracy, feed_dict={x:valid_features, y:valid_labels, keep_prob: 1.0})
    print("Cost: {0:.2f}   Accuracy: {1:.2f}%  " .format(result_cost,result_accuracy*100), end='')
    return result_cost, result_accuracy

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        #print(labels[start:end])
        yield features[start:end], labels[start:end]

def select_dataset(training):
    if LOCAL_DATASET == True:
        path = './labeled_images'
        dataset = 'labeled_images'
    elif training is True:
        path = './mnist'
        dataset = 'training'
    else:
        path = './mnist'
        dataset = 'testing'
    return path, dataset

def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    path, dataset = select_dataset(training = True)
    data = dataset_lib.get_data(batch_id, dataset=dataset, path=path)
    features = [np.array(x[1]) for x in data]
    labels = np.array([x[0] for x in data])

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)

def load_preprocess_testset(batch_id=None):
    batch_id = random.randint(1,5) if batch_id == None else batch_id
    path, dataset = select_dataset(training = False)
    data = dataset_lib.get_data(batch_id, dataset=dataset, path=path)
    features = [np.array(x[1]) for x in data]
    labels = np.array([x[0] for x in data])
    return features, labels



def train_network(nconv1, nconv2, nfullyconn, nfullyconn2, epochs, batch_size, keep_probability):
    from time import time
    x, y, keep_prob, cost, optimizer, accuracy = build_network(nconv1, nconv2, nfullyconn, nfullyconn2)
    valid_features, valid_labels = load_preprocess_testset()
    
    #print('Checking the Training on a Single Batch...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())
        start_time = time()
        project_start = start_time
        # Training cycle
        for epoch in range(epochs):
            batch_i = 1
            for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels, x, y, keep_prob)
            
            if epoch == 0 or (epoch+1)%25 == 0:
                print('Epoch {:>2}, EMNIST Batch {}:  '.format(epoch + 1, batch_i), end='')
                result_cost, result_accuracy = print_stats(
                                                    sess, 
                                                    batch_features, 
                                                    batch_labels, 
                                                    cost, 
                                                    accuracy,
                                                    x, 
                                                    y, 
                                                    keep_prob, 
                                                    valid_features, 
                                                    valid_labels)
                print("Time lapse: {:.2f} minute(s)." .format((time() - start_time)/60))
                start_time = time()
            
            
        print('Epoch {:>2}, EMNIST Batch {}:  '.format(epoch + 1, batch_i), end='')
        result_cost, result_accuracy = print_stats(
                                                    sess, 
                                                    batch_features, 
                                                    batch_labels, 
                                                    cost, 
                                                    accuracy,
                                                    x, 
                                                    y, 
                                                    keep_prob, 
                                                    valid_features, 
                                                    valid_labels)
        total_times = (time() - project_start)/60
        print("Time Lapse: {:.2f} minute(s)." .format(total_times), end='\r')
        
    return  result_accuracy, total_times/epochs
    
def vector_2d(array):
    return np.array(array).reshape((-1, 1))

def gaussian_process(x_train, y_train, x_test):
    import warnings
    def fxn():
        warnings.warn("deprecated", DeprecationWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()
        x_train = vector_2d(x_train)
        y_train = vector_2d(y_train)
        x_test = vector_2d(x_test)
        #print(x_train)
        #print(y_train)
        #print(x_test)

        # Train gaussian process
        gp = GaussianProcess(corr='squared_exponential',
                             theta0=1e-1, thetaL=1e-3, thetaU=1)
        gp.fit(x_train, y_train)

        # Get mean and standard deviation for each possible
        # number of hidden units
        y_mean, y_var = gp.predict(x_test, eval_MSE=True)
        y_std = np.sqrt(vector_2d(y_var))

    return y_mean, y_std

def next_parameter_by_ei(y_max, y_mean, y_std, x_choices):
    # Calculate expecte improvement from 95% confidence interval
    y_highestexpected = (y_mean + 1.96 * y_std)
    expected_improvement = y_highestexpected - y_max
    expected_improvement[expected_improvement < 0] = 0

    max_index = expected_improvement.argmax()
    # Select next choice
    next_parameter = x_choices[max_index]

    return next_parameter
    
def plot_gp_bounds(x, y, x_predict, y_predict, y_std, ax=None):
    if ax is None:
        ax = plt.gca()
        
    bound1 = y_predict + 1.96 * y_std
    bound2 = y_predict - 1.96 * y_std

    ax.plot(x_predict, y_predict, color=colors[1])
    ax.plot(x_predict, bound1, color=colors[1])
    ax.plot(x_predict, bound2, color=colors[1])
    
    ax.fill_between(
        x_predict,
        bound1.reshape(len(bound1)),
        bound2.reshape(len(bound2)),
        alpha=0.3
    )
    ax.scatter(x, y, color=colors[1], s=50)
    
    return ax

def saveas(name):
    image_name = '{}.png'.format(name)
    image_path = os.path.join(CURRENT_DIR, image_name)
    plt.savefig(image_path, facecolor='#f8fafb', bbox_inches='tight')
    
def hyperparam_selection(func, param, stat,n_hidden_range, nconv2, nfullyconn2, func_args=None, n_iter=20, min_score=0):
    if func_args is None:
        func_args = []

    scores = []
    parameters = []

    min_n_hidden, max_n_hidden = n_hidden_range
    n_hidden_choices = np.arange(min_n_hidden, max_n_hidden + 1)
    n_hidden = random.randint(min_n_hidden, max_n_hidden)
    if param == 'nconv1':
        nconv1 = n_hidden
        nfullyconn = stat
    else:
        nconv1 = stat
        nfullyconn = n_hidden    

    # To be able to perform gaussian process we need to
    # have at least 2 samples.

    #print("Iteration: 1")
    if (nconv1 == nfullyconn or
           nconv1 == nfullyconn2 or
           nfullyconn == nfullyconn2 or
           nconv1 == nconv2 or 
           nconv2 == nfullyconn or
           nconv2 == nfullyconn2):
               score = 0
    else:
        acc,t = func(nconv1, nconv2, nfullyconn, nfullyconn2, *func_args)
        score = acc*100-t

    parameters.append(n_hidden)
    scores.append(score)

    n_hidden = random.randint(min_n_hidden, max_n_hidden)

    for iteration in range(2, n_iter + 1):
        #print("\nIteration: {}" .format(iteration))
        n_hidden = int(n_hidden)
        if param == 'nconv1':
            nconv1 = n_hidden
            nfullyconn = stat
        else:
            nconv1 = stat
            nfullyconn = n_hidden 
        if (nconv1 == nfullyconn or
           nconv1 == nfullyconn2 or
           nfullyconn == nfullyconn2 or
           nconv1 == nconv2 or 
           nconv2 == nfullyconn or
           nconv2 == nfullyconn2):
               score = 0
        else:
            acc,t = func(nconv1, nconv2, nfullyconn, nfullyconn2, *func_args)
            score = acc*100-t
        
        parameters.append(n_hidden)
        scores.append(score)
        
        
        #x_train = vector_2d(parameters)
        #y_train = vector_2d(scores)
        #x_test = vector_2d(n_hidden_choices)


        #y_min = max(scores)
        y_mean, y_std = gaussian_process(parameters, scores,
                                         n_hidden_choices)

        #n_hidden = next_parameter_by_ei(y_min, y_mean, y_std,
                                        #n_hidden_choices)
        #########
        # Calculate expecte improvement from 95% confidence interval
        #expected_improvement = y_min - (y_mean - 1.96 * y_std)
        #expected_improvement[expected_improvement < 0] = 0

        #max_index = expected_improvement.argmax()
        # Select next choice based on expected improvement
        #n_hidden = n_hidden_choices[max_index]
        
        y_max = max(scores)
        y_highestexpected = (y_mean + 1.96 * y_std)
        expected_improvement = y_highestexpected - y_max
        expected_improvement[expected_improvement < 0] = 0
        max_index = expected_improvement.argmax()
        #print("y_highestexpected: {}  max_index: {} ".format(y_highestexpected, max_index ))
        # Select next choice
        n_hidden = n_hidden_choices[max_index]
        '''
        # Build plots
        ax1 = plt.subplot2grid((n_iter, 2), (iteration - 1, 0))
        ax2 = plt.subplot2grid((n_iter, 2), (iteration - 1, 1))
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        order = np.argsort(parameters)


        plot_gp_bounds(x_train, y_train, n_hidden_choices, y_mean, y_std, ax1)
        ax1.scatter(parameters[-1], scores[-1], marker='*', s=300,
                    color=colors[0], zorder=10, label='Last step')
        ax1.set_title("Gaussian Process\nafter {} iterations".format(iteration))
        ax1.set_xlim(min_n_hidden - 10, max_n_hidden + 10)
        ax1.set_xlabel("Number of outputs")
        ax1.set_ylabel("Score")
        #ax1.set_ylim(min(scores)*.9, max(scores))
        ax1.legend(loc='upper right')
        
        ax2.plot(n_hidden_choices, expected_improvement)
        ax2.scatter(n_hidden_choices[max_index], expected_improvement[max_index],
                    marker='*', s=300, color=colors[1], label='Next step', zorder=10)
        ax2.set_xlim(min_n_hidden - 10, max_n_hidden + 10)
        ax2.set_title("Expected Improvement\nafter {} iterations".format(iteration))
        ax2.set_xlabel("Number of outputs for {} layer".format(param))
        #ax2.set_ylim(-1, y_max + 10)
        
        if n_hidden > 450:
            ax2.legend(loc='upper left')
        else:
            ax2.legend(loc='upper right')
        
        '''
        if y_max == 100 or n_hidden in parameters or score < min_score:
            # Lowest expected improvement value have been achieved
            break

    max_score_index = np.argmax(scores)
    #for i in range(len(scores)):
        #print("Parameter: {} Score: {}" .format(parameters[i],scores[i]))
    return parameters[max_score_index], round(scores[max_score_index],2)
    
    
    
def hyperparam_wslist(scores, parameters, n_hidden_range):
    min_n_hidden, max_n_hidden = n_hidden_range
    n_hidden_choices = np.arange(min_n_hidden, max_n_hidden + 1)
    # To be able to perform gaussian process we need to
    # have at least 2 samples.
    if len(scores) < 2:
        return False

    #print("Iteration: 1")


    y_mean, y_std = gaussian_process(parameters, scores,
                                     n_hidden_choices)
    
    #n_hidden = next_parameter_by_ei(y_min, y_mean, y_std,
                                    #n_hidden_choices)
    #########
    # Calculate expecte improvement from 95% confidence interval
    #expected_improvement = y_min - (y_mean - 1.96 * y_std)
    #expected_improvement[expected_improvement < 0] = 0
    
    #max_index = expected_improvement.argmax()
    # Select next choice based on expected improvement
    #n_hidden = n_hidden_choices[max_index]
    
    y_max = max(scores)
    y_highestexpected = (y_mean + 1.96 * y_std)
    expected_improvement = y_highestexpected - y_max
    expected_improvement[expected_improvement < 0] = 0
    max_index = expected_improvement.argmax()

    return int(n_hidden_choices[max_index])

def findings_search(findings, hyparam_name):
    scores = []
    parameters = [] 
    sorted_dict = {}
    for score in findings:
        parameter = findings[score][hyparam_name]
        if parameter in sorted_dict:
            #print("Enter")
            if score > sorted_dict[parameter]['score']:
                #print("Enter")
                sorted_dict[parameter] = {'score': score}
        else:
            sorted_dict[parameter] = {'score': score}
            
    for parameter in sorted_dict:
        parameters.append(parameter)
        scores.append(sorted_dict[parameter]['score'])

    return scores, parameters

def elect_value(parameter, findings, valuerange):
    scores, parameters = findings_search(findings, parameter)
    if len(scores) >= 2:
        return hyperparam_wslist(scores, parameters, valuerange)
    else:
        return random.randint(valuerange[0],valuerange[1])
    

def optimize_network():
    epochs = 1
    batch_size = 518
    keep_probability = 0.8
    
    
    plt.close("all")
    best_n_hidden = random.randint(400,500)
    lastbest = best_n_hidden
    #lastparam = 'nfullyconn'
    param = ''
    print("Start")
    findings = {}
    max_score = 0
    for _ in range(1,5,1):

        
        nfullyconn2 = elect_value('nfullyconn2', findings, [50,200])
        nconv2 = elect_value('nconv2', findings, [10, 100])


        best_n_hidden = random.randint(30,200)
        lastbest = best_n_hidden
        for iteration in range(3):
            if iteration%2 ==0:
                param = 'nconv1'
                notparam = 'nfullyconn'
                #r = random.randint(30,500)
                #param_range = [min(r-50, 30), r+50]
                param_range = [40,90]
            else:
                param= 'nfullyconn'
                notparam = 'nconv1'
                ##r = random.randint(200,1000)
                #param_range = [min(max(r - 50 + lastbest, 30), 900), min(r + 50 + lastbest, 1000)]
                param_range = [400,500]
            #fig = plt.figure(figsize=(12, 16))
            best_n_hidden,score = hyperparam_selection(
                train_network,
                param,
                best_n_hidden,
                n_hidden_range=param_range,
                nconv2=nconv2,
                nfullyconn2=nfullyconn2,
                func_args=[epochs, batch_size, keep_probability],
                n_iter=6,
                min_score=max_score*.8
            )
            if score > max_score:
                max_score = score
            #plt.tight_layout()
            #plt.show()
            print("")
            findings[score] = {param: best_n_hidden, 
                               notparam: lastbest, 
                               'keep_prob': keep_probability, 
                               'nfullyconn2': nfullyconn2,
                               'nconv2': nconv2}
            lastbest = best_n_hidden
            for key in findings:
                print("{} {}".format(key,findings[key]))
        #saveas('hyperparam-selection-nn-hidden-units_' + str(keep_probability))
    
    nconv1 = findings[max(findings)]['nconv1']
    nconv2 = findings[max(findings)]['nconv2']
    nfullyconn = findings[max(findings)]['nfullyconn']
    nfullyconn2 = findings[max(findings)]['nfullyconn2']
    keep_probability = findings[max(findings)]['keep_prob']
    
    #saveas('hyperparam-selection-nn-hidden-units')
    return nconv1, nconv2, nfullyconn, nfullyconn2, keep_probability    
    
    
    
def train_full_model(nconv1, nconv2, nfullyconn, nfullyconn2, keep_probability):
    epochs = 5
    batch_size = 518
    keep_probability = 0.3
    
    valid_features, valid_labels = load_preprocess_testset()
    x, y, keep_prob, cost, optimizer, accuracy = build_network(nconv1, nconv2, nfullyconn, nfullyconn2)
    print('Training...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())
        start_time = time()
        project_start = start_time
        # Training cycle
        for epoch in range(epochs):
            for batch_i in range(1, 2):
                for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
                    train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels, x, y, keep_prob)
                
                if epoch == 0 or (epoch+1)%1 == 0:
                    print('Epoch {:>2}, EMNIST Batch {}:  '.format(epoch + 1, batch_i), end='')
                    result_cost, result_accuracy = print_stats(
                                                        sess, 
                                                        batch_features, 
                                                        batch_labels, 
                                                        cost, 
                                                        accuracy,
                                                        x, 
                                                        y, 
                                                        keep_prob, 
                                                        valid_features, 
                                                        valid_labels)
                    print("Time lapse: {:.2f} minute(s)." .format((time() - start_time)/60))
                    start_time = time()
            
            
        print('Epoch {:>2}, EMNIST Batch {}:  '.format(epoch + 1, batch_i), end='')
        result_cost, result_accuracy = print_stats(
                                                    sess, 
                                                    batch_features, 
                                                    batch_labels, 
                                                    cost, 
                                                    accuracy,
                                                    x, 
                                                    y, 
                                                    keep_prob, 
                                                    valid_features, 
                                                    valid_labels)
        total_times = (time() - project_start)/60
        print("Time Lapse: {:.2f} minute(s)." .format(total_times), end='\r')
        
        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, SAVE_MODEL_PATH)  
        return save_path


def test_model(valid_features, valid_labels):
    """
    Test the saved model against the test dataset
    """

    #valid_features, valid_labels = load_preprocess_testset()
    loaded_graph = tf.Graph()
    
    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(SAVE_MODEL_PATH + '.meta')
        loader.restore(sess, SAVE_MODEL_PATH)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        

        acc = sess.run(
                loaded_acc,
                feed_dict={loaded_x: valid_features, loaded_y: valid_labels, loaded_keep_prob: 1.0})

        return acc

def TF_model(image):
    loaded_graph = tf.Graph()
    print("Loading model...")
    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(SAVE_MODEL_PATH + '.meta')
        loader.restore(sess, SAVE_MODEL_PATH)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        #loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        #loaded_pred = loaded_graph.get_tensor_by_name('pred:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')

        feed_dict = {loaded_x: image, loaded_keep_prob: 1.0}
        classification = sess.run(tf.nn.top_k(tf.nn.softmax(loaded_logits), 1), feed_dict)
        return dataset_lib.map_to_ascii(classification.indices)

        
    
#nconv1, nconv2, nfullyconn, nfullyconn2, keep_probability = optimize_network()
#path = train_full_model(nconv1, nconv2, nfullyconn, nfullyconn2, keep_probability)

#average_acc = test_model()

#print('Testing Accuracy: {}\n'.format(average_acc))

#features = np.zeros((28,28,1), dtype=int)
#print(features)
#TF_model([features])
