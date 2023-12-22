import os, sys
from os.path import exists
from os import system

import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.stats
from scipy.stats import rankdata

np.set_printoptions(threshold=sys.maxsize)

##############################################################################




##############################################################################
## System Paths ##
path                 = './dataset/'
parameters           = {'0': 'sample.txt'} # Dictionary can be expanded for multiple test parameters

## Run Parameters ##
TEST_NUM_SET         = [0] # List can be expanded in case of multiple test parameters
best_model_path_list = ['./DeepCas9_Final/']

# Model
length = 30
class DeepCas9(tf.keras.Model):
    def __init__(self, filter_size, filter_num, node_1=80, node_2=60, l_rate=0.005):
        super(DeepCas9, self).__init__()
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.node_1 = node_1
        self.node_2 = node_2

        # Convolutional Layers - Renaming to match .h5 file
        self.conv1_w = tf.keras.layers.Conv2D(filters=filter_num[0], kernel_size=(1, filter_size[0]), padding='same', name='conv1_W')
        self.conv1_b = tf.keras.layers.Conv2D(filters=filter_num[0], kernel_size=(1, filter_size[0]), padding='same', use_bias=False, name='conv1_b')  # Assuming no bias used in conv1
        self.conv2_w = tf.keras.layers.Conv2D(filters=filter_num[1], kernel_size=(1, filter_size[1]), padding='same', name='conv2_W')
        self.conv2_b = tf.keras.layers.Conv2D(filters=filter_num[1], kernel_size=(1, filter_size[1]), padding='same', use_bias=False, name='conv2_b')  # Assuming no bias used in conv2
        self.conv3_w = tf.keras.layers.Conv2D(filters=filter_num[2], kernel_size=(1, filter_size[2]), padding='same', name='conv3_W')
        self.conv3_b = tf.keras.layers.Conv2D(filters=filter_num[2], kernel_size=(1, filter_size[2]), padding='same', use_bias=False, name='conv3_b')  # Assuming no bias used in conv3

        # Dropout Layer
        self.dropout = tf.keras.layers.Dropout(0.3)  # Dropout layers typically don't need naming

        # Fully Connected Layers
        self.dense1 = tf.keras.layers.Dense(node_1, activation='relu', name='Fully_Connected_Layer1')
        self.dense2 = tf.keras.layers.Dense(node_2, activation='relu', name='Fully_Connected_Layer2')
        self.output_layer = tf.keras.layers.Dense(1, name='Output_Layer')


        # Optimizer
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)

    def call(self, inputs, training=False):
        x = tf.reshape(inputs, [-1, 1, inputs.shape[-2], 4])  # Reshape inputs to match expected shape

        # First Conv Layer - Updated to use renamed layers
        x = self.conv1_w(x)
        x = tf.nn.relu(x)
        # x = self.conv1_b(x)  # Uncomment if bias is separately applied
        x = self.dropout(x, training=training)
        x = tf.nn.avg_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

        # Second Conv Layer - Updated to use renamed layers
        x = self.conv2_w(x)
        x = tf.nn.relu(x)
        # x = self.conv2_b(x)  # Uncomment if bias is separately applied
        x = self.dropout(x, training=training)
        x = tf.nn.avg_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

        # Third Conv Layer - Updated to use renamed layers
        x = self.conv3_w(x)
        x = tf.nn.relu(x)
        # x = self.conv3_b(x)  # Uncomment if bias is separately applied
        x = self.dropout(x, training=training)
        x = tf.nn.avg_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

        # Flatten and Fully Connected Layers
        x = tf.keras.layers.Flatten()(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            loss = tf.reduce_mean(tf.square(y - y_pred))

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Return a dictionary mapping metric names to current value
        return {'loss': loss}
    #def end: def __init__
#class end: DeepCas9

def Model_Finaltest(sess, TEST_X, filter_size, filter_num, if3d, model, args, load_episode, best_model_path):
    test_batch      = 500
    test_spearman   = 0.0
    optimizer       = model.optimizer
    TEST_Z          = np.zeros((TEST_X.shape[0], 1), dtype=float)
    
    for i in range(int(np.ceil(float(TEST_X.shape[0])/float(test_batch)))):
        Dict = {model.inputs: TEST_X[i*test_batch:(i+1)*test_batch], model.is_training: False}
        TEST_Z[i*test_batch:(i+1)*test_batch] = sess.run([model.outputs], feed_dict=Dict)[0]
    
    OUT = open("RANK_final_{}.txt".format(best_model_path.split('/')[1]), "a")
    OUT.write("Testing final \n {} ".format(tuple(TEST_Z.reshape([np.shape(TEST_Z)[0]]))))
    OUT.write("\n")
    OUT.close()
    return
#def end: Model_Finaltest


def preprocess_seq(data):
    print("Start preprocessing the sequence done 2d")
    length  = 30
    
    DATA_X = np.zeros((len(data),1,length,4), dtype=int)
    print(np.shape(data), len(data), length)
    for l in range(len(data)):
        for i in range(length):

            try: data[l][i]
            except: print(data[l], i, length, len(data))

            if data[l][i]in "Aa":    DATA_X[l, 0, i, 0] = 1
            elif data[l][i] in "Cc": DATA_X[l, 0, i, 1] = 1
            elif data[l][i] in "Gg": DATA_X[l, 0, i, 2] = 1
            elif data[l][i] in "Tt": DATA_X[l, 0, i, 3] = 1
            else:
                print("Non-ATGC character " + data[l])
                print(i)
                print(data[l][i])
                sys.exit()
        #loop end: i
    #loop end: l
    print("Preprocessing the sequence done")
    return DATA_X
#def end: preprocess_seq


def getseq(filenum):
    param   = parameters['%s' % filenum]
    FILE    = open(path+param, "r")
    data    = FILE.readlines()
    data_n  = len(data) - 1
    seq     = []

    for l in range(1, data_n+1):
        try:
            data_split = data[l].split()
            seq.append(data_split[1])
        except:
            print(data[l])
            seq.append(data[l])
    #loop end: l
    FILE.close()
    processed_full_seq = preprocess_seq(seq)

    return processed_full_seq, seq
#def end: getseq


#TensorFlow config
conf                                = tf.compat.v1.ConfigProto()
conf.gpu_options.allow_growth       = True
os.environ['CUDA_VISIBLE_DEVICES']  = '0'
best_model_cv                       = 0.0
best_model_list                     = []

for best_model_path in best_model_path_list:
    for modelname in os.listdir(best_model_path):
        if "meta" in modelname:
            best_model_list.append(modelname[:-5])
#loop end: best_model_path

TEST_X          = []
TEST_X_nohot    = []
for TEST_NUM in TEST_NUM_SET:
    tmp_X, tmp_X_nohot = getseq(TEST_NUM)
    TEST_X.append(tmp_X)
    TEST_X_nohot.append(tmp_X_nohot)
#loop end: TEST_NUM


for index in range(len(best_model_list)):
    best_model_path = best_model_path_list[index]
    best_model = best_model_list[index]
    valuelist = best_model.split('-')
    fulllist = []

    # Convert string values to their appropriate types
    for value in valuelist:
        if value == 'True': value = True
        elif value == 'False': value = False
        else:
            try: value = int(value)
            except ValueError:
                try: value = float(value)
                except ValueError: pass
        fulllist.append(value)

    # Parse model parameters
    if fulllist[2:][-3] is True:
        if3d, filter_size_1, filter_size_2, filter_size_3, filter_num_1, filter_num_2, filter_num_3, l_rate, load_episode, inception, node_1, node_2 = fulllist[2:]
        filter_size = [filter_size_1, filter_size_2, filter_size_3]
        filter_num  = [filter_num_1, filter_num_2, filter_num_3]
    else:
        if3d, filter_size, filter_num, l_rate, load_episode, inception, node_1, node_2 = fulllist[2:]

    args = [filter_size, filter_num, l_rate, load_episode]

    # Initialize and build the model
    model = DeepCas9(filter_size, filter_num, node_1, node_2, args[2])
    #dummy_input = tf.random.normal([1, 1, length, 4])
    dummy_input = tf.random.normal([1, length, 4])
    _ = model(dummy_input)
    model.compile()

    # Printing out the layer names and output shapes
    for layer in model.layers:
        print(layer.name, layer.output_shape)
        #print(layer.name)

    # Load the model weights
    model.load_weights('/home/yc774/ondemand/OOD/cellinfinity/Paired-Library/DeepCas9/DeepCas9_Final/model.weights.h5')

    # Process each test set
    for i in range(len(TEST_NUM_SET)):
        print(f"TEST_NUM: {TEST_NUM_SET[i]}")
        
        # Ensure Model_Finaltest is adapted for TensorFlow 2.x
        Model_Finaltest(TEST_X[i], filter_size, filter_num, if3d, model, args, load_episode, best_model_path)

