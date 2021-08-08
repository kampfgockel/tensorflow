# %%
# From the Udemy Deep Learning with TensorFlow 2.0 [2021] Course
# https://uofu.udemy.com/course/machine-learning-with-tensorflow-for-business-intelligence

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# %% Data Generation

observations = 1000
xs = np.random.uniform(low=-10 , high=10, size=(observations,1))
zs = np.random.uniform(low=-10 , high=10, size=(observations,1))

generated_inputs = np.column_stack((xs, zs))

noise = zs = np.random.uniform(low=-1 , high=1, size=(observations,1))

# You wouldn't normally know this. These match the weights and biases at line 59~
generated_targets = 2*xs - 3*zs + 5 + noise  

np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)


# %% Solving with TensorFlow
training_data = np.load('TF_intro.npz')

# %% Creating the Model and Fitting

input_size = 2  # There are two input variables x's and z's
output_size = 1 # Only one output y

model = tf.keras.Sequential([# Dense takes the inputs provided to the model and 
                            # calculates the dot product (np.dot(inputs,weights))
                            # and adds bias (np.dot(inputs,weights) + bias)
                            tf.keras.layers.Dense(output_size
                                                , kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1) # Kernel here means weight
                                                , bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1) # These are the randomly generated Biases
                                                )
                          
                            ])
# if you want to use a custom optimizer instead of the SGD optimizer below this is the same other than being able to set the learning rate ourselves
custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)

model.compile(
  optimizer=custom_optimizer # Using custom optimization algorithm
, loss='mean_squared_error' # Applying built-in loss function mean_squared_error (check out other options)
) 

# model.compile(
#     optimizer='sgd' # Using the Stochastic gradient descent optimization algorithm
#     , loss='mean_squared_error' # Applying built-in loss function mean_squared_error (check out other options)
# ) 


model.fit(
    training_data['inputs']  # Descriptors
  , training_data['targets'] # Response Variable
  , epochs = 100 # Epoch = Iteration over the full Data Set
  , verbose = 2 # 0 - Silent or no output 1 - full output
  )

# Review the results. The Loss should be decreasing meaning the model is working as intended

# %% Extract the Weights and Bias for a specific layer (0)

model.layers[0].get_weights() # Gets Weights and Biases


# [array([[1.9645482 ],                   <---- Weights Array roughly 2
#        [0.04933998]], dtype=float32),   
#  array([5.026239], dtype=float32)]      <---- Biases Array roughly 5

model.layers[0].get_weights()[0] # Gets Weights 
model.layers[0].get_weights()[1] # Gets Biases 

# This is precisely the information that confirms that our algorithm has learned the underlying relationship
# The relationship manually created earlier was 2*xs - 3*zs + 5 + noise but you wouldn't know this of course
# The goal of the algorithm is to figure this out

# %% Extract the outputs (predict values with our model)
model.predict_on_batch(training_data['inputs']).round(1) # calculate the outputs given inputs and round to one decimal point
# These are the outputs after the train model or outputs after 100 epochs of training
# Outputs are compared with the target at each epoch (iteration over full data set)

# %% compare the targets manually notice outputs above and targets below are similar but not the exact same
training_data['targets'].round(1)

# %% Plotting the Data This line should be as close to 45 degrees as possible since Targets should be close to outputs
plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()
# %%
