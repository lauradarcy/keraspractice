# Keras Tutorial Practice

also good github/readme practice

notes go here

Completed tutorial links:

https://www.tensorflow.org/tutorials/keras/basic_classification

https://www.tensorflow.org/tutorials/keras/basic_text_classification

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 16)          160000    
_________________________________________________________________
global_average_pooling1d (Gl (None, 16)                0         
_________________________________________________________________
dense (Dense)                (None, 16)                272       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 17        
=================================================================
Total params: 160,289
Trainable params: 160,289
Non-trainable params: 0
_________________________________________________________________
```

questions about this:

```(None, None, 16)``` , resulting dimensions are ```(batch, sequence, embedding)``` <-- what is this shape supposed to mean? why is ```None``` part of the shape?

## ```What the duck @ this whole thing```

also learned a new thing for pyplot: 

```
plt.figure()
fig = plt.gcf()
fig.canvas.set_window_title('Training and Validation Accuracy') #sets window title
```

## Basic Regression: notes

current tutorial link: https://www.tensorflow.org/tutorials/keras/basic_regression