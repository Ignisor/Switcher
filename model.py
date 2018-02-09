from keras.models import Sequential
from keras import layers
import settings
from utils import text_to_ids


class Switcher(object):
    def __init__(self):
        self.model = None
        self.vocabulary_size = settings.VOCABULARY_SIZE
        self.max_length = settings.PHRASE_MAX_LENGTH

    def init_model(self):
        self.model = Sequential([
            layers.Embedding(self.vocabulary_size, self.max_length * 2, input_length=self.max_length),
            layers.Flatten(),
            layers.Dense(128),
            layers.Dense(32),
            layers.Dense(1, activation='sigmoid'),
        ])

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        self.load_weights()

    def save_weights(self):
        self.model.save_weights(self.get_weights_path())

    def load_weights(self):
        """Loads model weights if file exists"""
        try:
            self.model.load_weights(self.get_weights_path())
            print('Loaded weights')
            return True
        except OSError:
            return False

    def get_weights_path(self):
        return f'saved/{self.__class__.__name__}.h5'

    def predict(self, text):
        return self.model.predict(text_to_ids(text))
