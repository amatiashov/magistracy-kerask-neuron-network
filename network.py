# from __future__ import absolute_import, division, print_function, unicode_literals
import uuid
import logging
import numpy as np
from constants import *

from tensorflow.keras import backend as K
from keras_preprocessing.image import load_img
from keras_preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model


logging.basicConfig(
    format=u'%(threadName)s\t%(filename)s\t[LINE:%(lineno)d]# %(levelname)-8s\t [%(asctime)s]  %(message)s',
    level="INFO")

log = logging.getLogger(__name__)

log.info("Tensorflow version: %s" % tf.__version__)

img_width = 500
img_height = 500

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


log.info("loading train batches...")
train_batches = ImageDataGenerator().flow_from_directory(directory=os.path.join(RESOURCES_DIR, "batches", "train"),
                                                         target_size=(img_width, img_height),
                                                         class_mode="binary",
                                                         batch_size=1)

log.info("loading test batches...")
test_batches = ImageDataGenerator().flow_from_directory(directory=os.path.join(RESOURCES_DIR, "batches", "test"),
                                                        target_size=(img_width, img_height),
                                                        class_mode="binary",
                                                        batch_size=1)
print(train_batches.labels)
print(train_batches.filenames)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(generator=train_batches,
                    steps_per_epoch=train_batches.n,
                    validation_data=train_batches,
                    validation_steps=train_batches.n,
                    epochs=20)


test_loss, test_acc = model.evaluate(test_batches, verbose=1)

log.info("Точность на проверочных данных: %s" % str(test_acc))


if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
model.save(os.path.join(MODEL_DIR, "%s.h5" % str(uuid.uuid4())))


log.debug("loading test image...")
image_path = os.path.join(BATHES_DIR, "validate_image.jpg")
img = load_img(image_path, target_size=(500, 500, 3))
# plt.imshow(img)
img = np.expand_dims(img, axis=0)
result = model.predict_classes(img)
print(result[0])

image_path = os.path.join(BATHES_DIR, "validate_image.png")
img = load_img(image_path, target_size=(500, 500, 3))
# plt.imshow(img)
img = np.expand_dims(img, axis=0)
result = model.predict_classes(img)
print(result[0])
