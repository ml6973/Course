################### using the following command: 
# python tenssorflow_hidden.py --train ./intro_to_ann.csv --test ./intro_to_ann2.csv --num_epochs 1000


import tensorflow.python.platform

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
# Global variables.
NUM_LABELS = 2    # The number of labels.
BATCH_SIZE = 100  # The number of training examples to use per training step.
verbose = False

# tf.app.flags.DEFINE_string('train', None,
                           # 'File containing the training data (labels & features).')
# tf.app.flags.DEFINE_string('test', None,
                           # 'File containing the test data (labels & features).')
# tf.app.flags.DEFINE_integer('num_epochs', 1,
                            # 'Number of passes over the training data.')
# tf.app.flags.DEFINE_integer('num_hidden', 5,
                            # 'Number of nodes in the hidden layer.')
# tf.app.flags.DEFINE_boolean('verbose', False, 'Produce verbose output.')
# FLAGS = tf.app.flags.FLAGS

# Extract numpy representations of the labels and features given rows consisting of:
#   label, feat_0, feat_1, ..., feat_n
def extract_data(filename):

    # Arrays to hold the labels and feature vectors.
    labels = []
    fvecs = []

    # print "Hello here!!"

    # Iterate over the rows, splitting the label from the features. Convert labels
    # to integers and features to floats.
    for line in file(filename):
        row = line.split(",")
        # print row
        labels.append(int(row[2]))
        fvecs.append([float(x) for x in row[0:2]])
        # print row[0:2]

    # Convert the array of float arrays into a numpy float matrix.
    fvecs_np = np.matrix(fvecs).astype(np.float32)

    # Convert the array of int labels into a numpy array.
    labels_np = np.array(labels).astype(dtype=np.uint8)

    # Convert the int numpy array into a one-hot matrix.
    labels_onehot = (np.arange(NUM_LABELS) == labels_np[:, None]).astype(np.float32)

    # Return a pair of the feature matrix and the one-hot label matrix.
    return fvecs_np,labels_onehot

# Init weights method. (Lifted from Delip Rao: http://deliprao.com/archives/100)
def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))


# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
# def plot_decision_boundary(pred_func,X,y):
#     # Set min and max values and give it some padding
#     x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#     y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#     h = 0.01
#     # Generate a grid of points with distance h between them
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     # Predict the function value for the whole gid
#     Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     # Plot the contour and training examples
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

def pred_func(x,w_hidden,b_hidden,w_out,b_out):
    hidden = tf.nn.tanh(tf.matmul(tf.cast(x,tf.float32),w_hidden) + b_hidden)
    y = tf.nn.softmax(tf.matmul(tf.cast(hidden,tf.float32), w_out) + b_out)
    # print x
    print "w_hidden b_hidden w_out b_out"
    print w_hidden.eval()
    print b_hidden.eval()
    print w_out.eval()
    print b_out.eval()
    pred_result = tf.argmax(y,1)
    return pred_result.eval()
    


    
def main(argv=None):
    # Be verbose?
    # verbose = FLAGS.verbose
    
    # Get the data.
    # train_data_filename = FLAGS.train
    # test_data_filename = FLAGS.test

    train_data_filename = "intro_to_ann.csv"
    test_data_filename = "intro_to_ann2.csv"

    # Extract it into numpy arrays.
    train_data,train_labels = extract_data(train_data_filename)
    test_data, test_labels = extract_data(test_data_filename)

    # Get the shape of the training data.
    train_size,num_features = train_data.shape

    # Get the number of epochs for training.
    # num_epochs = FLAGS.num_epochs
    num_epochs =1000

    # Get the size of layer one.
    # num_hidden = FLAGS.num_hidden
    num_hidden = 5


 
    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, num_features])
    y_ = tf.placeholder("float", shape=[None, NUM_LABELS])

    
    # For the test data, hold the entire dataset in one constant node.
    test_data_node = tf.constant(test_data)

    # Define and initialize the network.

    # Initialize the hidden weights and biases.
    w_hidden = init_weights(
        [num_features, num_hidden],
        'xavier',
        xavier_params=(num_features, num_hidden))

    b_hidden = init_weights([1,num_hidden],'zeros')

    # The hidden layer.
    hidden = tf.nn.tanh(tf.matmul(x,w_hidden) + b_hidden)

    # Initialize the output weights and biases.
    w_out = init_weights(
        [num_hidden, NUM_LABELS],
        'xavier',
        xavier_params=(num_hidden, NUM_LABELS))
    
    b_out = init_weights([1,NUM_LABELS],'zeros')

    # The output layer.
    y = tf.nn.softmax(tf.matmul(hidden, w_out) + b_out)
    
    # Optimization.
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    # Evaluation.
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
    	tf.initialize_all_variables().run()
    	if verbose:
    	    print 'Initialized!'
    	    print
    	    print 'Training.'
    	    
    	# Iterate and train.
    	for step in xrange(num_epochs * train_size // BATCH_SIZE):
    	    if verbose:
    	        print step,
    	        
    	    offset = (step * BATCH_SIZE) % train_size
    	    batch_data = train_data[offset:(offset + BATCH_SIZE), :]
    	    batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
    	    train_step.run(feed_dict={x: batch_data, y_: batch_labels})
    	    if verbose and offset >= train_size-BATCH_SIZE:
    	        print
    	print "Accuracy:", accuracy.eval(feed_dict={x: test_data, y_: test_labels})
        
        train = pd.read_csv("intro_to_ann.csv")
        X, y = np.array(train.ix[:,0:2]), np.array(train.ix[:,2])
        
        # plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.BuGn)
        # print X
        # print y

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # print xx
        # Predict the function value for the whole gid
        Z = pred_func(np.c_[xx.ravel(), yy.ravel()],w_hidden,b_hidden,w_out,b_out)
        print Z
        Z = Z.reshape(xx.shape)

        # print Z.shape


        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.axis('off')
        
        plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.BuGn)
        plt.show()


            
if __name__ == '__main__':
    tf.app.run()