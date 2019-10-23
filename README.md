# retraining-xray-classifier
Demonstrating the principle of transfer learning on an XRay classifier.

This repo demonstrates the how to carry out, and the utility of, transfer learning. It creates a custom image classifier and save the weights, then simulates a different dataset through applying perturbations to the image dataset. The dataset being used in the Chest X-Ray Image Dataset, available [HERE](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). This repo comes with a Jupyter notebook that details the logic behind the various sections of code and will allow the user to try editing the arguments of the classifiers.

The first portion of the script handles imports , declares important variables, and handles different possible input shapes for the model.

```Python
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten, Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

batch_size = 16
im_height = 150
im_width = 150

# handling the different possible input shapes for the model
if keras.backend.image_data_format() == 'channels_first':
    inp_shape = (3, im_width, im_height)
else:
    inp_shape = (im_width, im_height, 3)
```

The next code section sets the image directories, create the image data generators, creates iterable dataset objects with the generators. There are two train generators and two test generators. This is because a separate yet similar dataset is simulated by making perturbations to the validation and test datasets, and the transformations simulate the various damages that could happen to an X-ray. We will save the weights of the initial model after training, and use it as an example of transfer learning by finetuning the model and training on the second, warped dataset.

```Python
train_dir = "/chest_xray/train"
val_dir = "/chest_xray/val"
test_dir = "/chest_xray/test"

# generate the data for both training and validation data
# when we instantiate the data generator we pass in transformations to use
# rescaling and flipping here
train_1_datagen = ImageDataGenerator(rescale=1. / 255)

train_2_datagen = ImageDataGenerator(rescale=1. /255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, rotation_range=30,
                                     width_shift_range=0., channel_shift_range=0.9, brightness_range=[0.5, 1.5])

test_datagen = ImageDataGenerator(rescale=1. /255)

test_datagen_2 = ImageDataGenerator(rescale=1. /255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, rotation_range=30,
                                     width_shift_range=0., channel_shift_range=0.9, brightness_range=[0.5, 1.5])

# after creating the objects flow from directory
# declare what directory to flow from, as well as image size and batch size
# class mode is binary here, either normal xray or not normal

# we could do "class_mode = binary" here, if so, be sure to make the final output 1 and not 2
train_generator_1 = train_1_datagen.flow_from_directory(train_dir, target_size=(im_width, im_height),
                                                    batch_size=batch_size)

test_generator_1 = test_datagen.flow_from_directory(test_dir, target_size=(im_width, im_height),
                                                    batch_size = batch_size)

train_generator_2 = train_2_datagen.flow_from_directory(val_dir, target_size=(im_width, im_height),
                                                    batch_size = batch_size)

test_generator_2 = test_datagen_2.flow_from_directory(test_dir, target_size=(im_width, im_height),
                                                    batch_size = batch_size)

```

Here we check to see how the images differ.

```Python
def image_show(image_generator):
    x, y = image_generator.next()
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 3
    for i in range(1, 10):
        # img = np.random.randint(10)
        image = x[i]
        fig.add_subplot(rows, columns, i)
        plt.imshow(image.transpose(0, 1, 2))
    plt.show()

image_show(train_generator_1)
image_show(train_generator_2)
```

![](C:\jupyter_notebooks\XRay Classification\xray1.png)

![](C:\jupyter_notebooks\XRay Classification\xray2.png)

Here we create the custom model.

```Python
def create_model():
    # first specify the sequential nature of the model
    model = Sequential()
    # second parameter is the size of the "window" you want the CNN to use
    # the shape of the data we are passing in, 3 x 150 x 150
    # last element is just the image in the series, others are pixel widths
    model.add(Conv2D(64, (3, 3), input_shape=(150, 150, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    # you'll need to flatten the data again if you plan on having Dense layers in the model,
    # as it needs a 1d unlike a 2d CNN
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='sigmoid'))
    # now compile the model, specify loss, optimization, etc
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the model, specify batch size, validation split and epochs
    return model

model = create_model()
```

This section fits and trains the first model.

```Python
filepath = "weights_training_1.hdf5"
callbacks = [ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
              ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min', min_lr=0.00001),
              EarlyStopping(monitor= 'val_loss', min_delta=1e-10, patience=15, verbose=1, restore_best_weights=True)]

records = model.fit_generator(train_generator_1, steps_per_epoch=100, epochs=25, validation_data=test_generator_1, validation_steps=7, verbose=1, callbacks=callbacks)
```

Now we evaluate the first model's performance.

```Python
t_loss = records.history['loss']
v_loss = records.history['val_loss']
t_acc = records.history['acc']
v_acc = records.history['val_acc']

# gets the length of how long the model was trained for
train_length = range(1, len(t_acc) + 1)

def evaluation(model, train_length, training_acc, val_acc, training_loss, validation_loss, generator):

    # plot the loss across the number of epochs
    plt.figure()
    plt.plot(train_length, training_loss, label='Training Loss')
    plt.plot(train_length, validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(train_length, training_acc, label='Training Accuracy')
    plt.plot(train_length, val_acc, label='Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # compare against the test training set
    # get the score/accuracy for the current model
    scores = model.evaluate_generator(generator)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

evaluation(model, train_length, t_acc, v_acc, t_loss, v_loss, test_generator_1)
```

Here's what I got when evaluating the first model.

![](C:\jupyter_notebooks\XRay Classification\Figure_1.png)

![](C:\jupyter_notebooks\XRay Classification\Figure_2.png)

```
acc: 78.25%
```

This section creates and compiles the second model, training only the last three layers.

```Python
model_2 = create_model()
model_2.load_weights("weights_training_1.hdf5")

for layer in model_2.layers[:-5]:
    layer.trainable = False

for layer in model_2.layers:
    print(layer, layer.trainable)

# now compile the model, specify loss, optimization, etc
model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the model, specify batch size, validation split and epochs
```

Finally, the second model is fit/trained and evaluated.

```Python
filepath = "weights_training_2.hdf5"
callbacks = [ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
              ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min', min_lr=0.00001),
              EarlyStopping(monitor= 'val_loss', min_delta=1e-10, patience=15, verbose=1, restore_best_weights=True)]

records = model_2.fit_generator(train_generator_2, steps_per_epoch=90, epochs=20, validation_data=test_generator_2, validation_steps=7, verbose=1, callbacks=callbacks)

t_loss = records.history['loss']
v_loss = records.history['val_loss']
t_acc = records.history['acc']
v_acc = records.history['val_acc']

# gets the length of how long the model was trained for
train_length = range(1, len(t_acc) + 1)

evaluation(model_2, train_length, t_acc, v_acc, t_loss, v_loss, 80, test_generator_2)

```

Here were the results after evaluating the second model:

![](C:\jupyter_notebooks\XRay Classification\Figure_3.png)

![](C:\jupyter_notebooks\XRay Classification\Figure_4.png)

​	acc: 73.48%


We can see that after being trained for only a few epochs, and despite the perturbations applied to the dataset, performance is already around 73%.
