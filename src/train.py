import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from categorias import get_categorias
from medios import get_capturas_df

df = get_capturas_df()
categorias = get_categorias(df)


import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from medios import get_medios_path

batch_size = 32
img_height = 200  # 450
img_width = 200

# TODO: pre-proceso de distintos tamaños


medios_path = get_medios_path()
print(medios_path)

import pathlib

data_dir = pathlib.Path(medios_path)

image_count = len(list(data_dir.glob("*/*.jpg")))
print(image_count)
categoria = categorias.iloc[0]
print(categoria["Nombre"])
import random

imagen = list(data_dir.glob(f'{categoria["Categoria"]}/*'))
imagen_int = random.randint(0, len(imagen))

print(imagen[imagen_int])
PIL.Image.open(str(imagen[imagen_int]))

categoria = categorias.iloc[2]
print(categoria["Nombre"])
import random

imagen = list(data_dir.glob(f'{categoria["Categoria"]}/*'))
imagen_int = random.randint(0, len(imagen))

print(imagen[imagen_int])
PIL.Image.open(str(imagen[imagen_int]))
print("------------------------------------------------------")
print(f"loading datasests from {data_dir=}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

class_names = train_ds.class_names
print(f"----------->{class_names=}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(2):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(1000).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1.0 / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

model = Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.summary()

epochs = 10
print(f"running for {epochs=}")

history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()
