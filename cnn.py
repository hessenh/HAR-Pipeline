import tensorflow as tf
import numpy as np
import TRAINING_VARIABLES

V = TRAINING_VARIABLES.VARS()

class Convolutional_Neural_Network(object):
  def __init__(self):
    self._info = "Convolutional neural network with two convolutional layers and a fully connected network"

    '''Config'''
    self._input_size = V.CNN_INPUT_SIZE
    self._output_size = V.CNN_OUTPUT_SIZE
    self._iteration_size = V.CNN_NUMBER_OF_ITERATIONS
    self._batch_size = V.CNN_BATCH_SIZE
    self._model_name = V.CNN_MODEL_NAME
  
    filter_x = V.CNN_FILTER_X
    filter_y = V.CNN_FILTER_Y
    resize_y = V.CNN_RESIZE_Y
    resize_x = V.CNN_RESIZE_X
    window = V.CNN_WINDOW

    kernel_list = V.CNN_KERNEL_LIST
    number_of_kernels = V.CNN_NUMBER_OF_KERNELS
    connections_in = V.CNN_CONNECTIONS_INN
    neural_list = V.CNN_NEURAL_LIST
    FILTER_TYPE = V.CNN_FILTER_TYPE

    
    

    '''Placeholders for input and output'''
    self.x = tf.placeholder("float", shape=[None, self._input_size])
    #print self.x.get_shape(),'Input' 
    self.y_ = tf.placeholder("float", shape=[None, self._output_size])
    #print self.y_.get_shape(), 'Output'
    self.reshaped_input = tf.reshape(self.x, [-1, resize_y, resize_x, 1])
    #print self.reshaped_input.get_shape(), 'Input reshaped'


    def get_conv_layer(input_variables, number_kernels_in, number_kernels_out, layer_number, filter_x, filter_y, filter_type):
      weights = self.weight_variable([filter_y, filter_x, number_kernels_in, number_kernels_out], self._model_name + "w_conv_" + layer_number)
      bias = self.bias_variable([number_kernels_out],self._model_name + 'b_conv_' + layer_number)
      return tf.nn.relu(self.conv2d(input_variables, weights, filter_type) + bias)



    def connect_conv_layers(input_variables):
      
      output = get_conv_layer(input_variables, kernel_list[0], kernel_list[1], '1', filter_x, filter_y, FILTER_TYPE)
      #print output.get_shape(), 'Features 1'
      
      for i in range(1,len(kernel_list)-1):
        output = get_conv_layer(output, kernel_list[i], kernel_list[i+1], str(i+1), filter_x, filter_y, FILTER_TYPE)
        #print output.get_shape(), 'Features ', i+1

      # Flatten output
      output = tf.reshape(output, [-1, resize_y * (resize_x - (number_of_kernels*filter_x) + number_of_kernels) * kernel_list[-1]])
      return output    
    

    def get_nn_layer(input_variables, connections_in, connections_out, layer_number):
      weights = self.weight_variable([connections_in, connections_out], self._model_name + 'w_fc_' + str(layer_number+1))
      bias = self.bias_variable([connections_out], self._model_name +'b_fc_' + str(layer_number+1))
      return tf.nn.relu(tf.matmul(input_variables, weights) + bias)

    

    def connect_nn_layers(input_variables, keep_prob):
      output = get_nn_layer(input_variables, neural_list[0], neural_list[1],0)
      #print output.get_shape(), 'NN',0
      for i in range(1, len(neural_list)-2):
        output = get_nn_layer(output, neural_list[i], neural_list[i+1], i)
        print output.get_shape(),'NN',i


      output = tf.nn.dropout(output, keep_prob)
      # Last layer
      weights = self.weight_variable([neural_list[-2], neural_list[-1]], self._model_name + 'w_fc_last')
      bias = self.bias_variable([neural_list[-1]],self._model_name + 'b_fc_last')
      y_conv = tf.nn.softmax(tf.matmul(output, weights) + bias)
      return y_conv


    ''' Convolutinal layers'''
    self.output_conv = connect_conv_layers(self.reshaped_input)

    self.keep_prob = tf.placeholder("float")
    '''Densly conected layers'''
    self.y_conv = connect_nn_layers(self.output_conv, self.keep_prob)


    self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(tf.clip_by_value(self.y_conv,1e-10,1.0)))
    self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
    self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

    self.init_op = tf.initialize_all_variables()

  def weight_variable(self,shape,name):
    #print "weight_variable", shape
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name= name)

  def bias_variable(self, shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)

  def conv2d(self, x, W, filter_type):
    #print 'conv2d', x
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=filter_type)

  def max_pool_2x2(self, x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


  def set_data_set(self, data_set):
    self._data_set = data_set

  def load_model(self):
    self.sess = tf.Session()
    model = V.CNN_MODEL_PATH
    # Get last name. Path could be modles/test, so splitting on "/" and retreiving "test"
    model_name = model.split('/')
    model_name = model_name[-1]
    all_vars = tf.all_variables()
    model_vars = [k for k in all_vars if k.name.startswith(model_name)]
    tf.train.Saver(model_vars).restore(self.sess, model + '.ckpt')

  def save_model(self):
    path = V.CNN_MODEL_PATH
    saver = tf.train.Saver()
    save_path = saver.save(self.sess, path + '.ckpt')
    print("Model saved in file: %s" % save_path)


  ''' Train network '''
  def train_network(self):
    '''Creating model'''
    self.sess = tf.Session()
    self.sess.run(self.init_op)
    for i in range(self._iteration_size):
      batch = self._data_set.next_batch(self._batch_size)
      self.sess.run(self.train_step, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
  
  def get_accuracy(self):
    activities = V.ACTIVITIES

    length_of_data = len(self._data_set._data)
    #print length_of_data
    total_accuracy = 0.0
    total_accuracy_whole = 0.0
    for activity in activities:
      #step = length_of_data / 10
      
      activity_boolean = self._data_set._labels[::,activity] == 1.0
      activity_data = self._data_set._data[activity_boolean]
      activity_label = self._data_set._labels[activity_boolean]
      #print len(activity_data)
      length_of_temp_step = len(activity_data) / 10
      temp_score = 0.0
      for i in range(0, 10):
        temp_data = activity_data[i*length_of_temp_step:i*length_of_temp_step+length_of_temp_step]
        temp_label = activity_label[i*length_of_temp_step:i*length_of_temp_step + length_of_temp_step]
        temp_score += self.sess.run(self.accuracy,feed_dict={
          self.x: temp_data, self.y_: temp_label, self.keep_prob: 1.0})
      accuracy = temp_score / 10
      print str(accuracy).replace(".",",")
      total_accuracy += accuracy
      total_accuracy_whole += accuracy * (len(activity_data)*1.0/length_of_data)

    print str(total_accuracy_whole).replace(".",",")
    print str(total_accuracy / len(activities)).replace(".",",")




      

  def get_viterbi_data(self, data_set, number_of_samples):

    if len(data_set._data) < number_of_samples:
      number_of_samples = len(data_set._data)

    predictions = np.zeros((number_of_samples, len(V.ACTIVITIES)))
    actual = data_set._labels[0:number_of_samples]
    data_sensor = data_set._data[0:number_of_samples]
   

    for i in range(0, len(data_sensor)):
      data_batch = data_sensor[i]
      predictions[i] = self.sess.run(self.y_conv, feed_dict={self.x: [data_batch],self.keep_prob:1.0})


    return actual, predictions

  def get_predictions(self):
    
    length = len(self._data_set._data)
    
    predictions = np.zeros((length,len(V.ACTIVITIES)))


    for i in range(0, length):
      data_batch = self._data_set._data[i]
      predictions[i] = self.sess.run(self.y_conv, feed_dict={self.x: [data_batch],self.keep_prob:1.0})
      if i % 100 == 0:
        print str(i) + '/' + str(length)

    np.savetxt(V.VITERBI_PREDICTION_PATH_TESTING, predictions, delimiter=",")
    return predictions