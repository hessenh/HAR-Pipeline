import tensorflow as tf
import numpy as np
import TRAINING_VARIABLES
from data import DataSet

'''
MODULE FUNCTIONS
'''


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, filter_type):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=filter_type)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_conv_layer(input_variables, kernels_in, kernels_out, layer_number, filter_x, filter_y, filter_type, name):
    weights_shape = [filter_y, filter_x, kernels_in, kernels_out]
    weights_name = name + "w_conv_" + layer_number
    weights = weight_variable(weights_shape, weights_name)

    bias_shape = [kernels_out]
    bias_name = name + 'b_conv_' + layer_number
    bias = bias_variable(bias_shape, bias_name)

    features = conv2d(input_variables, weights, filter_type) + bias
    return tf.nn.relu(features)


def get_nn_layer(input_variables, connections_in, connections_out, layer_number, name):
    weights_shape = [connections_in, connections_out]
    weights_name = name + 'w_fc_' + str(layer_number + 1)
    weights = weight_variable(weights_shape, weights_name)

    bias_shape = [connections_out]
    bias_name = name + 'b_fc_' + str(layer_number + 1)
    bias = bias_variable(bias_shape, bias_name)

    features = tf.matmul(input_variables, weights) + bias
    return tf.nn.relu(features)


'''
CONFIGURATION VARIABLES
'''

V = TRAINING_VARIABLES.VARS()

'''
CNN CLASS
'''


class ConvolutionalNeuralNetwork(object):
    def __init__(self):
        self._info = "Convolutional neural network with two convolutional layers and a fully connected network"

        model_name = V.CNN_MODEL_NAME

        filter_x = V.CNN_FILTER_X
        filter_y = V.CNN_FILTER_Y
        resize_y = V.CNN_RESIZE_Y
        resize_x = V.CNN_RESIZE_X

        kernel_list = V.CNN_KERNEL_LIST
        number_of_kernels = V.CNN_NUMBER_OF_KERNELS
        neural_list = V.CNN_NEURAL_LIST
        filter_type = V.CNN_FILTER_TYPE

        def connect_conv_layers(input_variables):
            output = input_variables

            for i in range(0, len(kernel_list) - 1):
                output = get_conv_layer(output,
                                        kernel_list[i],
                                        kernel_list[i + 1],
                                        str(i + 1),
                                        filter_x,
                                        filter_y,
                                        filter_type,
                                        model_name)

            output_shape = resize_y * (resize_x - (number_of_kernels * filter_x) + number_of_kernels) * kernel_list[-1]
            flatten_output = tf.reshape(output, [-1, output_shape])
            output = flatten_output

            return output

        def connect_nn_layers(input_variables, keep_prob):
            output = input_variables

            for i in range(0, len(neural_list) - 2):
                output = get_nn_layer(output,
                                      neural_list[i],
                                      neural_list[i + 1],
                                      i,
                                      model_name)
                print(output.get_shape(), 'NN', i)

            # Last layer
            output = tf.nn.dropout(output, keep_prob)
            weights = weight_variable([neural_list[-2], neural_list[-1]], model_name + 'w_fc_last')
            bias = bias_variable([neural_list[-1]], model_name + 'b_fc_last')
            y_conv = tf.nn.softmax(tf.matmul(output, weights) + bias)

            return y_conv

        def make_session():
            session = tf.Session()
            session.run(tf.initialize_all_variables())
            return session

        '''
        CNN INSTANCE VARIABLES
        '''

        '''Model configurations'''
        self._input_size = V.CNN_INPUT_SIZE
        self._output_size = V.CNN_OUTPUT_SIZE
        self._iteration_size = V.CNN_NUMBER_OF_ITERATIONS
        self._batch_size = V.CNN_BATCH_SIZE
        self._model_name = V.CNN_MODEL_NAME

        '''Tensorflow placeholders'''
        self._x = tf.placeholder("float", shape=[None, self._input_size])
        self._y = tf.placeholder("float", shape=[None, self._output_size])

        '''Convolutinal layers'''
        self._output_conv = connect_conv_layers(tf.reshape(self._x, [-1, resize_y, resize_x, 1]))

        '''Densly conected layers'''
        self._keep_prob = tf.placeholder("float")
        self._y_conv = connect_nn_layers(self._output_conv, self._keep_prob)

        '''Tensorflow variables'''
        self._cross_entropy = -tf.reduce_sum(self._y * tf.log(tf.clip_by_value(self._y_conv, 1e-10, 1.0)))
        self._train_step = tf.train.AdamOptimizer(1e-4).minimize(self._cross_entropy)
        self._correct_prediction = tf.equal(tf.argmax(self._y_conv, 1), tf.argmax(self._y, 1))
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, "float"))

        '''Data containers'''
        self._data_set = DataSet([], [])
        self._session = make_session()

    '''
    CNN CLASS METHODS
    '''

    def set_data_set(self, data_set):
        self._data_set = data_set

    def load_model(self):
        load_path = V.CNN_MODEL_PATH
        model_name = load_path.split('/')[-1]

        all_vars = tf.all_variables()
        model_vars = [k for k in all_vars if k.name.startswith(model_name)]

        tf.train.Saver(model_vars).restore(self._session, load_path + '.ckpt')
        print("Model loaded from %s" % load_path)

    def save_model(self):
        path = V.CNN_MODEL_PATH
        saver = tf.train.Saver()
        save_path = saver.save(self._session, path + '.ckpt')
        print("Model saved in file: %s" % save_path)

    def train_network(self):
        for i in range(self._iteration_size):
            batch = self._data_set.next_batch(self._batch_size)
            self._session.run(self._train_step, feed_dict={self._x: batch[0],
                                                           self._y: batch[1],
                                                           self._keep_prob: 0.5})

    def get_accuracy(self):
        activities = V.ACTIVITIES

        total_accuracy = 0.0
        total_accuracy_whole = 0.0

        for activity in activities:
            activity_boolean = self._data_set.labels[::, activity] == 1.0
            activity_data = self._data_set.data[activity_boolean]
            activity_label = self._data_set.labels[activity_boolean]

            length_of_temp_step = len(activity_data) / 10
            temp_score = 0.0

            for i in range(0, 10):
                temp_data = activity_data[i * length_of_temp_step:i * length_of_temp_step + length_of_temp_step]
                temp_label = activity_label[i * length_of_temp_step:i * length_of_temp_step + length_of_temp_step]
                temp_score += self._session.run(self._accuracy, feed_dict={self._x: temp_data,
                                                                           self._y: temp_label,
                                                                           self._keep_prob: 1.0})

            accuracy = temp_score / 10
            print str(accuracy).replace(".", ",")

            total_accuracy += accuracy
            total_accuracy_whole += accuracy * (len(activity_data) * 1.0 / len(self._data_set.data))

        print str(total_accuracy_whole).replace(".", ",")
        print str(total_accuracy / len(activities)).replace(".", ",")

    def get_predictions(self):
        data_batch = self._data_set.data
        predictions = self._session.run(self._y_conv, feed_dict={self._x: data_batch,
                                                                 self._keep_prob: 1.0})
        return predictions

    def get_viterbi_data(self, data_set, number_of_samples):

        if len(data_set.data) < number_of_samples:
            number_of_samples = len(data_set.data)

        actual = data_set.labels[0:number_of_samples]
        data_sensor = data_set.data[0:number_of_samples]

        # predictions = np.zeros((len(data_sensor), len(V.ACTIVITIES)))
        predictions = self._session.run(self._y_conv, feed_dict={self._x: data_sensor, self._keep_prob: 1.0})
        # memory = 10
        # length_of_temp_step = len(predictions) / memory
        # for i in range(0, memory):
        #  data_batch = data_sensor[i*length_of_temp_step:i*length_of_temp_step+length_of_temp_step]
        #  predictions[i*length_of_temp_step:i*length_of_temp_step+length_of_temp_step] =
        # self.sess.run(self.y_conv, feed_dict={self.x: data_batch,self.keep_prob:1.0})

        return actual, predictions
