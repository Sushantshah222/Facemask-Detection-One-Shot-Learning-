

from check import *
import cv2
from tensorflow import keras
import numpy as np
import tensorflow as tf



def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)


def liveimg_to_encoding(image, model):
    img = np.around(np.array(image) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)


def check_mask(image, database, model):
    encoding = img_to_encoding(image, model)

    min_dist = 100

    for (name, db_enc) in database.items():

        dist = np.linalg.norm(db_enc - encoding)

        if dist < min_dist:
            min_dist = dist
            identity = name

    return min_dist, identity


def check_mask_live(image, database, model):
    encoding = liveimg_to_encoding(image, model)

    min_dist = 100

    for (name, db_enc) in database.items():

        dist = np.linalg.norm(db_enc - encoding)

        if dist < min_dist:
            min_dist = dist
            identity = name

    return min_dist, identity