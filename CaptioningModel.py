import pickle
import numpy as np
from itertools import groupby
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input, InceptionV3

path_prefix = '/Users/jagan/Downloads/HPC Project/'
train_images_dir = path_prefix + 'Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt'
test_images_dir = path_prefix + 'Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt'
images_dir = path_prefix + 'Flickr_Data/Flickr_Data/Images/'
model_dir = path_prefix + 'data/model_199.h5'
wordtoix_dir = path_prefix + 'data/wordtoix.pkl'
ixtoword_dir = path_prefix + 'data/ixtoword.pkl'


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def get_train_images():
    train_images = list()
    doc = load_doc(train_images_dir)

    for line in doc.split('\n'):
        if len(line) < 1:
            continue

        identifier = line.split('.')[0]
        train_images.append(images_dir + identifier + '.jpg')

    return train_images


def get_test_images():
    test_images = list()
    doc = load_doc(test_images_dir)

    for line in doc.split('\n'):
        if len(line) < 1:
            continue

        identifier = line.split('.')[0]
        test_images.append(images_dir + identifier + '.jpg')

    return test_images


def preprocess_image(filename):
    model = InceptionV3(weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = load_img(filename, target_size=(299, 299))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return np.reshape(feature, (1, feature.shape[1]))


class CaptioningModel:
    start_token = 'Start '
    end_token = ' End'
    max_length = 30

    def __init__(self):
        self.train_images = get_train_images()
        self.test_images = get_test_images()
        self.wordtoix = pickle.load(open(wordtoix_dir, 'rb'))
        self.ixtoword = pickle.load(open(ixtoword_dir, 'rb'))

    def greedySearch(self, photo):
        model = load_model(model_dir)
        in_text = self.start_token[:-1]

        for i in range(15):
            sequence = [self.wordtoix[w] for w in in_text.split() if w in self.wordtoix]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            yhat = model.predict([photo, sequence])
            yhat = np.argmax(yhat)
            word = self.ixtoword[yhat]
            in_text += ' ' + word

            if word == self.end_token[1:]:
                break

        final = in_text.split()
        final = final[1:-1]
        final = [x[0] for x in groupby(final)]
        final = ' '.join(final)
        return final

    def predict(self, image_dir):
        image = preprocess_image(image_dir)
        return self.greedySearch(image)
