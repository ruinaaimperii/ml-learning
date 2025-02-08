import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.image import imread


# eventually it didn't work (fix later)
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy so cancelling training")
        self.model.stop_training = True


callbacks = myCallback()

fashion = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion.load_data()
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')])

test_check = int(input("Which test are we doing? 1/2 "))
if test_check == 1:
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])
    model.summary()
    model.fit(training_images, training_labels, epochs=5)

    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_accuracy}', f'Test loss: {test_loss*100}%')
    f, axarr = plt.subplots(3, 4)
    image_index1 = 0
    image_index2 = 23
    image_index3 = 28
    convolution_number = 6
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)
    for x in range(0, 4):
        f1 = activation_model.predict(test_images[image_index1].reshape(1, 28, 28, 1))[x]
        axarr[0, x].imshow(f1[0, :, :, convolution_number], cmap='inferno')
        axarr[0, x].grid(False)
        f2 = activation_model.predict(test_images[image_index2].reshape(1, 28, 28, 1))[x]
        axarr[1, x].imshow(f2[0, :, :, convolution_number], cmap='inferno')
        axarr[1, x].grid(False)
        f3 = activation_model.predict(test_images[image_index3].reshape(1, 28, 28, 1))[x]
        axarr[2, x].imshow(f3[0, :, :, convolution_number], cmap='inferno')
        axarr[2, x].grid(False)
    plt.show()