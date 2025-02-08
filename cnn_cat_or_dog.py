import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile


# splitting the data
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero size, ignoring")

    training_length = int(len(files) * SPLIT_SIZE)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[:training_length]
    testing_set = shuffled_set[training_length:]

    for filename in training_set:
        current_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(current_file, destination)

    for filename in testing_set:
        current_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(current_file, destination)


CAT_SOURCE_DIR = 'datasets/PetImages/Cat/'
TRAINING_CAT_DIR = 'datasets/PetImages/Cat_trainig_set/'
TESTING_CAT_DIR = 'datasets/PetImages/Cat_test_set/'
DOG_SOURCE_DIR = 'datasets/PetImages/Dog/'
TRAINING_DOG_DIR = 'datasets/PetImages/Dog_training_set/'
TESTING_DOG_DIR = 'datasets/PetImages/Dog_test_set/'

SPLIT_SIZE = 0.9
split_data(CAT_SOURCE_DIR, TRAINING_CAT_DIR, TESTING_CAT_DIR, SPLIT_SIZE)
split_data(DOG_SOURCE_DIR, TRAINING_DOG_DIR, TESTING_DOG_DIR, SPLIT_SIZE)


# model =
# UNFINISHED
# UNFINISHED
# UNFINISHED
# UNFINISHED
