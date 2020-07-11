# Tensor Flow 2.1.0 Hello World in Azure Machine learning Service Compute Instance

## Build a simple Hello world tensor flow model to test tensor flow version 2.1.0

## First upgrade pip version to latest

```
!pip install --upgrade pip
```

## Now check the tensorflow version to make sure we have the correct version

```
import tensorflow as tf
print(tf.__version__)
```

If running into issues, please check the error for dependencies packages and do pip install them. At the time when i created a compute instance i had version 2.1.0

## To install tensorflow 

```
pip install --upgrade tensorflow
```

Here is where you can find the latest version of tensorflow packages.

https://www.tensorflow.org/install/pip

The above is to setup the necessary packages.

Now time to start the coding.

## Hello world sample

import tenforflow package.

```
import tensorflow as tf
```

Load data set and split for training and test.

```
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

Now build the deep neural network architecture

```
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```

Now train the model

```
predictions = model(x_train[:1]).numpy()
predictions
```

Convert to classes to probablities

```
tf.nn.softmax(predictions).numpy()
```

The losses.SparseCategoricalCrossentropy loss takes a vector of logits and a True index and returns a scalar loss for each example.

```
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.log(1/10) ~= 2.3.

```
loss_fn(y_train[:1], predictions).numpy()
```

```
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
```

The Model.fit method adjusts the model parameters to minimize the loss:

```
model.fit(x_train, y_train, epochs=5)
```

Evaulate the model with test data

```
model.evaluate(x_test,  y_test, verbose=2)
```

If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:

```
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
```

Display the output

```
probability_model(x_test[:5])
```

End of the sample.