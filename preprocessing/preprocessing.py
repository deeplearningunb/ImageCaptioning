import cv2
import glob
import tensorflow as tf

class Preprocessing():

    def __init__(self):
        # image related data
        self.images_name = []
        self.images_list = []

        # caption related data
        self.captions_list = []
        self.captions_mapped = {}
        self.vocab = {}

    def _load_and_preprocess_images(self):
        path = '../dataset/images/Flicker8k_Dataset/*.jpg'
        for img in glob.glob(path):
            img = img.split('/')[-1]
            self.images_name.append(img)
            image = cv2.imread(img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224,224))
            self.images_list.append(image)
    
    def _load_and_preprocess_captions(self):
        self._load_and_preprocess_images()
        f = open('../dataset/text/Flickr8k.token.txt', 'rb')
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

    def transform_caption_in_vocab(self):
        self._load_and_preprocess_captions()

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
    
    def extract_features(self):
        pass
