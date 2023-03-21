# Importing Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from medios import get_medios_polipos_path

# creating models directory
import os

BASE_DIR = "/home/elpacko/Gastroclub/"
# checking TensorFlow version and GPU usage
print('Tensorflow version:', tf.__version__)
print('Is using GPU?', tf.test.is_gpu_available())

main_dir = get_medios_polipos_path()
# Setting path to the training directory
train_dir = os.path.join(main_dir, 'train')

# Setting path to the test directory
test_dir = os.path.join(main_dir, 'test')

# Directory with train POLIP images
train_POLIP_dir = os.path.join(train_dir, 'POLIPO')

# Directory with train normal images
train_normal_dir = os.path.join(train_dir, 'NOPOLIPO')

# Directory with test POLIP image
test_POLIP_dir = os.path.join(test_dir, 'POLIPO')

# Directory with test normal image
test_normal_dir = os.path.join(test_dir, 'NOPOLIPO')

# Creating a list of filenames in each directory
train_POLIP_names = os.listdir(train_POLIP_dir)
print(train_POLIP_names[:10])  # printing a list of the first 10 filenames

train_normal_names = os.listdir(train_normal_dir)
print(train_normal_names[:10])

test_POLIP_names = os.listdir(test_POLIP_dir)
print(test_POLIP_names[:10])

test_normal_names = os.listdir(test_normal_dir)
print(test_normal_names[:10])

# Printing total number of images present in each set
print('Total no of images in training set:', len(train_POLIP_names
                                                + train_normal_names))
print("Total no of images in test set:", len(test_POLIP_names
                                            + test_normal_names))

# Data Visualization
import matplotlib.image as mpimg
# Setting the no of rows and columns
ROWS = 4
COLS = 4
# Setting the figure size
fig = plt.gcf()
# get current figure; allows us to get a reference to current figure when using pyplot
fig.set_size_inches(12, 12)


# get the directory to each image file in the trainset
POLIP_pic = [os.path.join(train_POLIP_dir, filename) for filename in train_POLIP_names[:8]]
normal_pic = [os.path.join(train_normal_dir, filename) for filename in train_normal_names[:8]]
print(POLIP_pic)
print(normal_pic)
# merge POLIP and normal lists
merged_list = POLIP_pic + normal_pic
print(merged_list)

# Plotting the images in the merged list
for i, img_path in enumerate(merged_list):
    # getting the filename from the directory
    data = img_path.split('/', 6)[6]
    # creating a subplot of images with the no. of rows and colums with index no
    sp = plt.subplot(ROWS, COLS, i+1)
    # turn off axis
    sp.axis('Off')
    # reading the image data to an array
    img = mpimg.imread(img_path)
    # setting title of plot as the filename
    sp.set_title(data, fontsize=6)
    # displaying data as image
    plt.imshow(img, cmap='gray')
    
plt.show()  # display the plot


# Data Preprocessing and Augmentation
# Generate training, testing and validation batches
dgen_train = ImageDataGenerator(rescale=1./255,
                                validation_split=0.2,  # using 20% of training data for validation 
                                zoom_range=0.2,
                                horizontal_flip=True)
dgen_validation = ImageDataGenerator(rescale=1./255)
dgen_test = ImageDataGenerator(rescale=1./255)

# Awesome HyperParameters!!!
TARGET_SIZE = (200, 200)
BATCH_SIZE = 32
CLASS_MODE = 'binary'  # for two classes; categorical for over 2 classes

# Connecting the ImageDataGenerator objects to our dataset
train_generator = dgen_train.flow_from_directory(train_dir,
                                                target_size=TARGET_SIZE,
                                                subset='training',
                                                batch_size=BATCH_SIZE,
                                                class_mode=CLASS_MODE)

validation_generator = dgen_train.flow_from_directory(train_dir,
                                                      target_size=TARGET_SIZE,
                                                      subset='validation',
                                                      batch_size=BATCH_SIZE,
                                                      class_mode=CLASS_MODE)
test_generator = dgen_test.flow_from_directory(test_dir,
                                              target_size=TARGET_SIZE,
                                              batch_size=BATCH_SIZE,
                                              class_mode=CLASS_MODE)

# Get the class indices
train_generator.class_indices

# Get the image shape
train_generator.image_shape

def trainmodel():
  # Building CNN Model
  model = Sequential()
  model.add(Conv2D(32, (5,5), padding='same', activation='relu',
                  input_shape=(200, 200, 3)))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.2))
  model.add(Conv2D(64, (5,5), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))
  model.summary()

  # Compile the Model
  model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

  # Train the Model
  history = model.fit(train_generator,
            epochs=30,
            validation_data=validation_generator,
            callbacks=[
            # Stopping our training if val_accuracy doesn't improve after 20 epochs
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                            patience=20),
            # Saving the best weights of our model in the model directory
          
            # We don't want to save just the weight, but also the model architecture
            tf.keras.callbacks.ModelCheckpoint('models/model_{val_accuracy:.3f}.h5',
                                            save_best_only=True,
                                            save_weights_only=False,
                                            monitor='val_accuracy'
                                              )
      ])



  history.history.keys()

  # Plot graph between training and validation loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.legend(['Training', 'Validation'])
  plt.title('Training and Validation Losses')
  plt.xlabel('epoch')

  # Plot graph between training and validation accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.legend(['Training', 'Validation'])
  plt.xlabel('epoch')

def testmodel(filename):
  # loading the best perfoming model
  model = tf.keras.models.load_model('models/model_0.762.h5')

  # Getting test accuracy and loss
  test_loss, test_acc = model.evaluate(test_generator)
  print('Test loss: {} Test Acc: {}'.format(test_loss, test_acc))

  # Making a Single Prediction
  import numpy as np
  from keras.preprocessing import image
  from keras.utils import load_img, img_to_array
  # load and resize image to 200x200
  test_image = load_img(os.path.join(BASE_DIR, 'new', filename),
                              target_size=(200,200))

  # convert image to numpy array
  images = img_to_array(test_image)
  # expand dimension of image
  images = np.expand_dims(images, axis=0)
  # making prediction with model
  prediction = model.predict(images)
      
  if prediction == 0:
    print('POLIP Detected')
  else:
    print('Report is Normal')

