import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt

print("\ntf version: ",tf.__version__)

#loading the data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#info on training images and labels
print("\n\ntrain_images.shape: ",train_images.shape)
print("len(train_labels): ",len(train_labels))
print("train_labels: ",train_labels)

#info on testing images and labels
print("\n\ntest_images.shape: ",test_images.shape)
print("len(test_labels): ",len(test_labels))
print("test_labels: ",test_labels)

#preprocess pixel vals from 0 to 255 to 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

#display figure for training image 0
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

#new figure
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], )#cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])


#building a model consisting of 3 layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #transforms 28*28 into a 1d array of 28*28=784 pixels, only a reformatting layer
    keras.layers.Dense(128, activation=tf.nn.relu), #128 neurons
    keras.layers.Dense(10, activation=tf.nn.softmax) #10 node softmax layer - returns an array of 10 probability scores that sum to 1, indicating prob of belonging to one of the 10 classes
])
#compile the model
model.compile(optimizer=tf.train.AdamOptimizer(),  #how the model is updated based on the data it sees and its loss function
              loss='sparse_categorical_crossentropy', #measures how accurate model is during training
              #want to minimize this function to "steer" the model in the right direction
              metrics=['accuracy']) #used to monitor the training and testing steps
#train the model:
model.fit(train_images, train_labels, epochs=5)
#test the model:
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

#make some predictions:
predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))

#graphing predictions
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)


'''
Finally, use the trained model to make a prediction about a single image
models are optimized to make predictions on a batch, or collection, of examples at once. 
So even though we're using a single image, we need to add it to a list:
'''
# Grab an image from the test dataset
img = test_images[0]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)
#predict the image
predictions_single = model.predict(img)

print(predictions_single)
plt.figure()
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()