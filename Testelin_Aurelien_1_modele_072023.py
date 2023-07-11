from flask import Flask, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

import numpy as np

app = Flask(__name__)

# Charger le tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Charger le modèle
model = load_model('LSTM_model.h5')

def prepare_keras_data(docs, max_sequence_length=40):
    encoded_docs = tokenizer.texts_to_sequences(docs)
    padded_docs = pad_sequences(encoded_docs, int(max_sequence_length), padding='post')
    return padded_docs


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    tweet = data['tweet']
    padded_sequences = prepare_keras_data(tweet)

    prediction = model.predict(np.array(padded_sequences))
    sentiment = np.argmax(prediction)

    return str(sentiment)