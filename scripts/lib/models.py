import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model

def leakyrelu(input, alpha=0.3):
  return tf.maximum(alpha * input, input)

def mlp_generator(input_shape, dim, output_size, num_layers=4):
    # Use tf.keras.Input to create a Keras tensor
    inputs = tf.keras.Input(shape=input_shape)
    
    x = inputs
    for layer in range(num_layers):
        in_size = input_shape[0] if layer == 0 else dim
        out_size = output_size if layer == num_layers - 1 else dim
        
        # Create layers
        x = layers.Dense(units=out_size, activation='relu',
                         kernel_initializer=tf.keras.initializers.RandomNormal(stddev=tf.sqrt(2. / in_size)))(x)
    
    outputs = x
    model = Model(inputs, outputs)
    return model


# def mlp_generator(inputs, dim, input_size, output_size, num_layers=4):
#     inputs = tf.reshape(inputs, [-1, input_size])
    
#     for layer in range(num_layers):
#         in_size = input_size if layer == 0 else dim
#         out_size = output_size if layer == num_layers - 1 else dim
        
#         # Weight initialization
#         W = tf.Variable(tf.random.truncated_normal([in_size, out_size], stddev=tf.sqrt(2. / tf.cast(in_size, tf.float32))),
#                         name='Layer_{}/Weights'.format(layer))
        
#         # Bias initialization
#         b = tf.Variable(tf.zeros([out_size]), name='Layer_{}/Bias'.format(layer))
        
#         # Apply the layer: inputs * W + b, followed by ReLU activation
#         inputs = tf.nn.relu(tf.add(tf.matmul(inputs, W), b))
    
#     outputs = inputs
#     return outputs

def mlp_discriminator(inputs, dim, input_size, num_layers=4):
    inputs = tf.reshape(inputs, [-1, input_size])  # Ensure inputs are reshaped properly

    for layer in range(num_layers):
        in_size = input_size if layer == 0 else dim
        out_size = 1 if layer == num_layers - 1 else dim
        
        # Weight initialization using tf.Variable
        W = tf.Variable(tf.random.truncated_normal(shape=[in_size, out_size], 
                                                   stddev=tf.sqrt(2. / tf.cast(in_size, tf.float32))),
                        name='Layer_{}/Weights'.format(layer))
        
        # Bias initialization using tf.Variable
        b = tf.Variable(tf.zeros([out_size]), name='Layer_{}/Bias'.format(layer))
        
        # Apply the layer: inputs * W + b, followed by ReLU activation
        inputs = tf.nn.relu(tf.matmul(inputs, W) + b)
    
    scores = inputs  # Final scores after the last layer
    return scores

def mlp_discriminator(input_shape, dim, num_layers=4):
    # Use tf.keras.Input to create a Keras tensor
    inputs = tf.keras.Input(shape=input_shape)
    
    x = inputs
    for layer in range(num_layers):
        in_size = input_shape[0] if layer == 0 else dim
        out_size = 1 if layer == num_layers - 1 else dim
        
        # Create a Dense layer
        x = layers.Dense(units=out_size, activation='relu',
                         kernel_initializer=tf.keras.initializers.RandomNormal(stddev=tf.sqrt(2. / in_size)))(x)
    
    scores = x  # Final scores after the last layer
    model = Model(inputs, scores)
    return model


def resblock(inputs, num_channels, name):
  input_size = tf.reduce_prod(tf.shape(inputs[1:]))
  res = inputs
  for r in range(2):
    res = tf.nn.relu(res)
    filters = tf.get_variable('Layer_{}/Resblock_{}/filters'.format(name, r),
                              initializer=tf.random_uniform(shape=[5, num_channels, num_channels],
                                                            minval=-tf.sqrt(3.) * tf.sqrt(4. / tf.cast(5 * num_channels + 5 * num_channels, tf.float32)),
                                                            maxval=tf.sqrt(3.) * tf.sqrt(4. / tf.cast(5 * num_channels + 5 * num_channels, tf.float32))))  # (width, in_chan, out_chan)    
    conv = tf.nn.conv1d(res, filters, stride=1, padding="SAME")  # (batch, width, out_chan)
    bias = tf.get_variable('Layer_{}/Resblock_{}/bias'.format(name, r), initializer=tf.constant(0.0, shape=[num_channels]))
    res = conv + bias
  return inputs + (0.3 * res)

def resnet_generator(inputs, num_channels, seq_len, vocab_size, annotated=False, res_layers=5):
  with tf.Session() as sess:
    input_size = sess.run(tf.shape(inputs)[1])
  output_size = seq_len * num_channels
  W = tf.get_variable('Layer_0/Linear/Weights',
                      #validate_shape=False,
                      initializer=tf.random_uniform(shape=[input_size, output_size], minval=-tf.sqrt(3.) * tf.sqrt(2. / tf.cast(input_size + output_size, tf.float32)), maxval=tf.sqrt(3.) * tf.sqrt(2. / tf.cast(input_size + output_size, tf.float32))))
  b = tf.get_variable('Layer_0/Linear/Bias', initializer=tf.constant(0.0, shape=[output_size]))
  outputs = tf.add(tf.matmul(inputs, W), b)
  inputs = tf.reshape(outputs, [-1, seq_len, num_channels])
  for layer in range(res_layers):
    outputs = resblock(inputs, num_channels, layer + 1)
    inputs = outputs
  filters = tf.get_variable('Layer_{}/filters'.format(layer + 2),
                            initializer=tf.random_uniform(shape=[1, num_channels, vocab_size],
                                                            minval=-tf.sqrt(3.) * tf.sqrt(4. / tf.cast(1 * num_channels + 1 * vocab_size, tf.float32)),
                                                            maxval=tf.sqrt(3.) * tf.sqrt(4. / tf.cast(1 * num_channels + 1 * vocab_size, tf.float32)))) #(width, in_chan, out_chan)
  conv = tf.nn.conv1d(outputs, filters, stride=1, padding="SAME") #(batch, width, out_chan)
  bias = tf.get_variable('Layer_{}/bias'.format(layer + 2), initializer=tf.constant(0.0, shape=[vocab_size]))
  output = conv + bias
  if annotated:
    logits = output[:,:,:-1]
    ann = tf.nn.sigmoid(tf.expand_dims(output[:,:,-1],2))
  else:
    logits = output
  probs = tf.nn.softmax(logits)

  if annotated:
    out = tf.concat([probs, ann], 2)
  else:
    out = probs
  return out
  

def resnet_discriminator(inputs, num_channels, seq_len, vocab_size, res_layers=5):
  input_size = seq_len * vocab_size
  filters = tf.get_variable('Layer_0/filters',
                            initializer=tf.random_uniform(shape=[1, vocab_size, num_channels],
                                                          minval=-tf.sqrt(3.) * tf.sqrt(4. / tf.cast(1 * num_channels + 1 * vocab_size, tf.float32)),
                                                          maxval=tf.sqrt(3.) * tf.sqrt(4. / tf.cast(1 * num_channels + 1 * vocab_size, tf.float32))))  # (width, in_chan, out_chan)
  conv = tf.nn.conv1d(inputs, filters, stride=1, padding="SAME")  # (batch, width, out_chan)
  bias = tf.get_variable('Layer_0/bias', initializer=tf.constant(0.0, shape=[num_channels]))
  inputs = conv + bias
  for layer in range(res_layers):
    outputs = resblock(inputs, num_channels, layer + 1)
    inputs = outputs

  inputs = tf.reshape(outputs, [-1, seq_len * num_channels])
  input_size = seq_len * num_channels
  W = tf.get_variable('Layer_{}/Linear/Weights'.format(layer + 2),
                      #validate_shape=False,
                      initializer=tf.random_uniform(shape=[input_size, 1], minval=-tf.sqrt(3.) * tf.sqrt(2. / tf.cast(input_size + 1, tf.float32)), maxval=tf.sqrt(3.) * tf.sqrt(2. / tf.cast(input_size + 1, tf.float32))))
  b = tf.get_variable('Layer_{}/Linear/Bias'.format(layer + 2), initializer=tf.constant(0.0, shape=[1]))
  scores = tf.add(tf.matmul(inputs, W), b)
  return scores