import settings
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.sequence import pad_sequences


def text_to_ids(text):
    if type(text) == str:
        text = [text]

    encoded_texts = [hashing_trick(t, settings.VOCABULARY_SIZE, hash_function='md5') for t in text]
    padded_texts = pad_sequences(encoded_texts, maxlen=settings.PHRASE_MAX_LENGTH, padding='post')

    return padded_texts
