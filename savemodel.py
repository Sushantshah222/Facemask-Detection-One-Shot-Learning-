from tensorflow.keras.applications.inception_v3 import InceptionV3

model = InceptionV3(input_shape=(299, 299, 3), include_top=True, weights='imagenet')
model.summary()
model.save("facemask_model.h5")