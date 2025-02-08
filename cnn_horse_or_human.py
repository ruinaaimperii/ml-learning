import os
import tensorflow as tf
import numpy as np


train_horse_dir = os.path.join('datasets/horse-or-human/horses')
train_human_dir = os.path.join('datasets/horse-or-human/humans')

model1 = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
                                     tf.keras.layers.MaxPooling2D(2, 2),
                                     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                     tf.keras.layers.MaxPooling2D(2, 2),
                                     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                     tf.keras.layers.MaxPooling2D(2, 2),
                                     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                     tf.keras.layers.MaxPooling2D(2, 2),
                                     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                     tf.keras.layers.MaxPooling2D(2, 2),
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(256, activation='relu'),
                                     tf.keras.layers.Dense(256, activation='relu'),
                                     tf.keras.layers.Dense(1, activation='relu')
                                     ])
model1.summary()
model1.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['auc', 'acc'])


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('datasets/horse-or-human/',
                                                    target_size=(300, 300),
                                                    batch_size=128,
                                                    class_mode='binary')

history = model1.fit(train_generator,
                     steps_per_epoch=8,
                     epochs=15,
                     verbose=1)

for name in os.listdir('datasets/horse-or-human/test_pics'):
    path = 'datasets/horse-or-human/test_pics/' + name
    img = tf.keras.preprocessing.image.load_img(path, target_size=(300, 300))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model1.predict(images)
    print(classes[0])
    if classes[0] > 0.5:
        print(name + ' is a human')
    else:
        print(name + ' is a horse')