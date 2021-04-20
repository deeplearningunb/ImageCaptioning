import cv2
import glob
import tensorflow as tf
import numpy as np
import pickle
import os

class Preprocessing():

    def __init__(self):
        # image related data
        self.images_name = []
        self.images_list = []
        self.images_features = {}
        self.img_features_path = os.path.join('./dataset/pkl/train2048.pkl')

        # caption related data
        self.captions_list = []
        self.captions_mapped = {}
        self.vocab = {}

        # encode model for preprocessing
        self.encode_model = None

    def _load_and_preprocess_images(self):
        if not os.path.exists(self.img_features_path):
            print("a")
            path = './dataset/images/Flicker8k_Dataset/*.jpg'
            for img in glob.glob(path):
                image = cv2.imread(img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (299,299))
                self.images_list.append(image)
                img = img.split('/')[-1]
                self.images_name.append(img)
        
    def _creating_cnn_model_for_feature_extraction(self):
        x = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet')
        self.encode_model = tf.keras.models.Model(x.input, x.layers[-2].output)
    
    def _prepare_and_feed_images_to_encode_model(self):
        if not os.path.exists(self.img_features_path):
            for j in range(0,len(self.images_list)):
                i = self.images_list[j].reshape(1,299,299,3)
                i = self.encode_model.predict(i).reshape(2048,)
                self.images_features[self.images_name[j]] = i

            with open(self.img_features_path, 'wb') as f:
                pickle.dump(self.images_features, f)
        
        else:
            print('File already exist, opening...')
            with open(self.img_features_path, 'rb') as f:
                self.images_features = pickle.load(f)


    def _load_and_preprocess_captions(self):
        f = open('./dataset/text/Flickr8k.token.txt', 'rb')
        self.captions_list = f.read().decode('utf-8').split('\n')
        
        for i in self.captions_list:
            name = i.split('\t')[0][:-2]
            caption = i.split('\t')[1]
            caption = caption.lower()
            caption = 'sequencestart ' + caption + ' sequenceend'
            
            if name in self.images_name:
                if name not in self.captions_mapped:
                    self.captions_mapped[name] = [caption]
                else:
                    self.captions_mapped[name].append(caption)

    def _transform_caption_in_vocab(self):
        mapped_number = 1
        for w in self.captions_mapped.values():
            for words in w:
                for word in words.split():
                    if word not in self.vocab:
                        self.vocab[word] = mapped_number
                        mapped_number += 1
        
        for i, w in self.captions_mapped.items():
            for words in w:
                mapped = []
                for word in words.split():
                    mapped.append(self.vocab[word])
            
            self.captions_mapped[i][w.index(words)] = mapped

    def generator(self):
        pass


    def preprocess_all(self):
        self._load_and_preprocess_images()
        self._creating_cnn_model_for_feature_extraction()
        self._prepare_and_feed_images_to_encode_model()
        self._load_and_preprocess_captions()
        self._transform_caption_in_vocab()


if __name__ == "__main__":
    p = Preprocessing()
    p.preprocess_all()