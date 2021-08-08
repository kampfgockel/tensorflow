#%%
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# %%
mnist_dataset, mnist_info = tfds.load(
                                      name='mnist'
                                    , with_info=True     # info on version, features, # of samples
                                    , as_supervised=True # Loads the data set as [input,target]
                                    )
# %%
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples # pulls the number of samples from the mnist_info variable could have also just counted them directly
num_validation_samples = tf.cast(num_validation_samples, tf.int64) # rounding down incase 10% isn't a clean number

num_test_samples = mnist_info.splits['test'].num_examples 
num_test_samples = tf.cast(num_test_samples, tf.int64)

# %%
def scale(image, label): # our transformation function
    image = tf.cast(image, tf.float32)
    image /= 255.   # the . means we want the output to be a float
    return image, label 

scaled_train_and_validation_data = mnist_train.map(scale) # takes a transformation function with two inputs and outputs

test_data = mnist_test.map(scale)

# %% Shuffle the data for batching so each batch has a random sample of the whole
BUFFER_SIZE = 10000 # If the buffer size is > n shuffling will happen all at once if buffer size is < n we optimize computational power

shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE) # actually shuffling and resaving the data

validation_data = shuffled_train_and_validation_data.take(num_validation_samples) # takes the # of samples previously specified
train_data = shuffled_train_and_validation_data.skip(num_validation_samples) # takes the rest to use as trining data

BATCH_SIZE = 100 # 1 = STOCHASTIC GRADIENT DESCENT, n = Single batch Gradient Descent, anything between 1 and n = mini batch Gradient Descent
train_data = train_data.batch(BATCH_SIZE)
# Note: Validation data doesn't need to be batched because we don't backward propogate but it still needs to be in batch form for the model to accept
validation_data = validation_data.batch(num_validation_samples) # Converting the validation data to one batch the size of itself
#test_data = ???
# iter makes it possible to iterate through the data, next loads next element. In this case there is only one batch so it'll load inputs and targets
validation_inputs, validation_targets = next(iter(validation_data)) 


# %%
input_size = 784
output_size = 10
hidden_layer_size = 100 # increased from 50 to 100 and validation accuracy increased 2%

model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape = (28,28,1)), # Input Layer
            tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # Hidden Layer 1 can be optimized by trying dif activation functions
            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),  # Hidden layer 2 (You can add as many as you want)
            tf.keras.layers.Dense(output_size, activation='softmax') # Output Layer 
            # The Output layer must use a activation function that converts hidden layer values to  probabilities (%) for output 
])
# %% Choose optimizer and loss function
model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# Optimizer
  # 'adam' (not case sensitive) = adaptive moment estimator 
  
# Loss Function
  # We want to use a loss function that is used for classifiers (cross entropy is usually the first choice)
  # Binary_crossentropy  = for cases where we have binary encoding
  # Categorical_crossentropy = expects that you've one-hot encoded the targets
  # Sparse_categorical_crossentropy = applies the on-hot encoding

# Metrics
  # Metrics we calculate throughout the training and testing processes
# %% Training The model
NUM_EPOCHS = 5 

model.fit(train_data  # model iterates over the whole training data set
        , epochs = NUM_EPOCHS  
        , validation_data = (validation_inputs, validation_targets) # forward propogate the whole validation dataset in a single batch
        , verbose = 2 # Receive only the most important info for each epoch
        )

# 540/540 - number of batches
# loss - training loss should be compared accross epochs should be decreasing
# accuracy - shows in what % of cases the output matches is equal to the targets
# val_loss - this is the check for overfitting 
# val_accuracy - true accuracy of the model for the epoch (accuracy of the whole validation set)
# %%
