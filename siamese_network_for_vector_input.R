# This file presents a simple implementation of siamese networks in Keras in R.
# It focuses solely on datapoints represented as feature vectors (e.g. Boston housing problem)
# as oppose to typical use case with images and convolutional layers.
#
# Here, I used Boston housing price regression which is a dataset readily available in Keras and
# sourced from the StatLib library which is maintained at the Carnegie Mellon University.
# Datapoints contain 13 features of houses in Boston in 1970s and the prices of the houses.
# In this script, the data is pre-processed to match the siamese problem definition with learning
# a similarity metric, i.e. the aim is to predict the difference in prices of 2 input houses.
#
# The training takes about 5 mins on a 4-core 4.2 GHz Intel i7 CPU
#
# by Kajetan Stanski, 8 Nov 2019


library(keras)


### Load and pre-process data ###
boston_housing <- dataset_boston_housing()

is_regression <- FALSE  # otherwise classification

similarity <- function(a, b) {
  s <- abs(a-b)
  if (is_regression) {
    return(s)
  }
  if (s < 7.2) {  # where 7.2 is the median price difference in the training dataset
    return(0)
  }
  return(1)
}

produce_dataset <- function(vectors, house_prices) {
  no_of_vectors <- dim(vectors)[1]
  dataset_size <- no_of_vectors * (no_of_vectors-1) / 2
  vectors_length <- dim(vectors)[2]
  x_right <- matrix(, nrow=dataset_size, ncol=vectors_length)
  x_left <- matrix(, nrow=dataset_size, ncol=vectors_length)
  y <- matrix(, nrow=dataset_size, ncol=1)
  datum_idx <- 1
  for (i in 1:(no_of_vectors-1)) {
    for (j in (i+1):no_of_vectors) {
      x_right[datum_idx, ] <- vectors[i, ]
      x_left[datum_idx, ] <- vectors[j, ]
      y[datum_idx, ] <- similarity(house_prices[i], house_prices[j])
      datum_idx <- datum_idx + 1
    }
  }
  return(list(list(x_right, x_left), y))
}

train_dataset <- produce_dataset(boston_housing$train$x, boston_housing$train$y)
validation_dataset <- produce_dataset(boston_housing$test$x, boston_housing$test$y)
input_vectors_length <- dim(boston_housing$train$x)[2]


### Build siamese network model ###
left_input_vector <- layer_input(shape=c(input_vectors_length), name='x_left')
right_input_vector <- layer_input(shape=c(input_vectors_length), name='x_right')

# Add regularization factor
if (is_regression) {
  reg_l2 <- 10^0
} else {
  reg_l2 <- 10^-2
}
no_of_layers <- 5
base_network <- keras_model_sequential(name='base_network')
hidden_layer <- base_network
for (i in 1:no_of_layers) {
  hidden_layer <- hidden_layer %>% layer_dense(units=input_vectors_length, activation='relu',
                                               name=paste('dense', i, sep=''),
                                               kernel_regularizer = regularizer_l2(l=reg_l2))
}

left_output <- left_input_vector %>% base_network
right_output <- right_input_vector %>% base_network

distance_l1 <- function(vectors) { 
  c(x,y) %<-% vectors
  return(k_abs(x-y))
}       

distance_layer <- layer_lambda(object = list(left_output, right_output), f=distance_l1, name='distance_layer')   

if (is_regression) {
  activation_function <- 'linear'
} else {
  activation_function <- 'sigmoid'
}
prediction <- distance_layer %>% layer_dense(units=1, activation=activation_function, name='y',
                                             kernel_regularizer = regularizer_l2(l=reg_l2))

model <- keras_model(list(left_input_vector, right_input_vector), prediction)
model %>% summary()


### Fit the model ###
if (is_regression) {
  model %>% compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=c('mae')
  )
} else {
  model %>% compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=c('accuracy')
  )
}

history <- fit(
  model,
  x=train_dataset[[1]],
  y=train_dataset[[2]],
  validation_data = validation_dataset,
  validation_steps = 10,
  steps_per_epoch = 100,
  epochs = 10
)

plot(history)


### Verify if predictions make sense ###
predicted_vs_observed <- function(x_a, y_a, x_b, y_b) {
  predicted_similarity <- model %>% predict(list(x_a, x_b))
  observed_similarity <- similarity(y_a, y_b)
  print(paste('predicted_similarity =', predicted_similarity))
  print(paste('observed_similarity =', observed_similarity))
}

row_idx_l <- 12  # arbitrary datapoint
row_idx_r <- 67  # arbitrary datapoint
x_left <- matrix(boston_housing$test$x[row_idx_l, ], nrow=1)
y_left <- boston_housing$test$y[row_idx_l]
x_right <- matrix(boston_housing$test$x[row_idx_r, ], nrow=1)
y_right <- boston_housing$test$y[row_idx_r]

print('Different vectors')
predicted_vs_observed(x_left, y_left, x_right, y_right)
print('Different vectors, swapped places (the output should be exactly the same)')
predicted_vs_observed(x_right, y_right, x_left, y_left)
print('The same vector')
predicted_vs_observed(x_left, y_left, x_left, y_left)

