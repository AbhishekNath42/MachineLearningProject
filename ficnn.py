# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu'))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(.25))

classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(.25))


classifier.add(Flatten())
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(.5))
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_set = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary')

classifier.fit_generator(
    train_set,
    steps_per_epoch=3893,
    epochs=15,
    validation_data=test_set,
    validation_steps=376)





classifier.save('cdi.h5')
classifierfile= classifier.to_json()
jsonfile= open('model.json','w')
jsonfile.write(classifier.to_json())
classifier.save_weights('cw.h5')

from keras.models import load_model
classifier = load_model('cdi.h5')

import numpy as np
from keras.preprocessing import image
import cv2

img = cv2.imread('1.jpg',0)
img2=cv2.imread('1.jpg')
img = cv2.resize(img,(48,48))
# print(img.shape)
img = image.img_to_array(img)
# print(img.shape)
img = np.expand_dims(img, axis=0)
result = classifier.predict(img)
# train_set.class_indices
if result[0][0] == 1:
    print('sad')
    cv2.putText(img2,'sad',(20, 100),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0), thickness=4)

else:
    print('happy')
    cv2.putText(img2, 'happy',(20, 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=4)

cv2.imshow('Result', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

