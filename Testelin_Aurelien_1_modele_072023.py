from flask import (Flask, redirect, render_template, request,
                   url_for, jsonify)

from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
import re
import spacy
from spacy.tokens import Doc
from spacy.language import Language
from sklearn.base import BaseEstimator, TransformerMixin
import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import numpy as np

app = Flask(__name__)
nlp = spacy.load('en_core_web_lg')
if 'expand_contractions' not in nlp.pipe_names:
    nlp.add_pipe('expand_contractions', before='tagger')

def setup():
    global tokenizer, interpreter

    # Charger le tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    model_path = "LSTM_model.tflite"
    if not os.path.isfile(model_path):
        download_model()

    # Charger le modèle
    interpreter = tf.lite.Interpreter(model_path="LSTM_model.tflite")
    interpreter.resize_tensor_input(input_index=interpreter.get_input_details()[0]['index'], tensor_size=[1, 40])
    interpreter.allocate_tensors()



def download_model():
    try:
        # Créer le client BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(
            "DefaultEndpointsProtocol=https;AccountName=tflitelstm;AccountKey=8GkilJ3mMDkQUWx6onXOE4+2q4ADNDZvH/aop9hCvOY0Iqup0uFluCxRg+zdLmkeAWJoE9RBScDI+AStCCvzWQ==;EndpointSuffix=core.windows.net")

        # Spécifiez le nom de votre conteneur et le nom du blob pour votre modèle tflite
        container_name = 'tflitecontainer'
        blob_name = 'LSTM_model.tflite'

        # Créer le BlobClient
        blob_client = blob_service_client.get_blob_client(container_name, blob_name)

        # Télécharger le blob en tant que fichier
        with open("LSTM_model.tflite", "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
    except Exception as ex:
        print('Exception:')
        print(ex)

def prepare_keras_data(docs, max_sequence_length=40):
    encoded_docs = tokenizer.texts_to_sequences(docs)
    padded_docs = pad_sequences(encoded_docs, int(max_sequence_length), padding='post')
    return padded_docs


def expand_contractions(text: str) -> str:
    flags = re.IGNORECASE | re.MULTILINE

    text = re.sub(r'`', "'", text, flags=flags)

    ## starts / ends with '
    text = re.sub(
        r"(\s|^)'(aight|cause)(\s|$)",
        '\g<1>\g<2>\g<3>',
        text, flags=flags
    )

    text = re.sub(
        r"(\s|^)'t(was|is)(\s|$)", r'\g<1>it \g<2>\g<3>',
        text,
        flags=flags
    )

    text = re.sub(
        r"(\s|^)ol'(\s|$)",
        '\g<1>old\g<2>',
        text, flags=flags
    )

    ## expand words without '
    text = re.sub(r"\b(aight)\b", 'alright', text, flags=flags)
    text = re.sub(r'\bcause\b', 'because', text, flags=flags)
    text = re.sub(r'\b(finna|gonna)\b', 'going to', text, flags=flags)
    text = re.sub(r'\bgimme\b', 'give me', text, flags=flags)
    text = re.sub(r"\bgive'n\b", 'given', text, flags=flags)
    text = re.sub(r"\bhowdy\b", 'how do you do', text, flags=flags)
    text = re.sub(r"\bgotta\b", 'got to', text, flags=flags)
    text = re.sub(r"\binnit\b", 'is it not', text, flags=flags)
    text = re.sub(r"\b(can)(not)\b", r'\g<1> \g<2>', text, flags=flags)
    text = re.sub(r"\bwanna\b", 'want to', text, flags=flags)
    text = re.sub(r"\bmethinks\b", 'me thinks', text, flags=flags)

    ## one offs,
    text = re.sub(r"\bo'er\b", r'over', text, flags=flags)
    text = re.sub(r"\bne'er\b", r'never', text, flags=flags)
    text = re.sub(r"\bo'?clock\b", 'of the clock', text, flags=flags)
    text = re.sub(r"\bma'am\b", 'madam', text, flags=flags)
    text = re.sub(r"\bgiv'n\b", 'given', text, flags=flags)
    text = re.sub(r"\be'er\b", 'ever', text, flags=flags)
    text = re.sub(r"\bd'ye\b", 'do you', text, flags=flags)
    text = re.sub(r"\be'er\b", 'ever', text, flags=flags)
    text = re.sub(r"\bd'ye\b", 'do you', text, flags=flags)
    text = re.sub(r"\bg'?day\b", 'good day', text, flags=flags)
    text = re.sub(r"\b(ain|amn)'?t\b", 'am not', text, flags=flags)
    text = re.sub(r"\b(are|can)'?t\b", r'\g<1> not', text, flags=flags)
    text = re.sub(r"\b(let)'?s\b", r'\g<1> us', text, flags=flags)

    ## major expansions involving smaller,
    text = re.sub(r"\by'all'dn't've'd\b", 'you all would not have had', text, flags=flags)
    text = re.sub(r"\by'all're\b", 'you all are', text, flags=flags)
    text = re.sub(r"\by'all'd've\b", 'you all would have', text, flags=flags)
    text = re.sub(r"(\s)y'all(\s)", r'\g<1>you all\g<2>', text, flags=flags)

    ## minor,
    text = re.sub(r"\b(won)'?t\b", 'will not', text, flags=flags)
    text = re.sub(r"\bhe'd\b", 'he had', text, flags=flags)

    ## major,
    text = re.sub(r"\b(I|we|who)'?d'?ve\b", r'\g<1> would have', text, flags=flags)
    text = re.sub(r"\b(could|would|must|should|would)n'?t'?ve\b", r'\g<1> not have', text, flags=flags)
    text = re.sub(r"\b(he)'?dn'?t'?ve'?d\b", r'\g<1> would not have had', text, flags=flags)
    text = re.sub(r"\b(daren|daresn|dasn)'?t", 'dare not', text, flags=flags)
    text = re.sub(r"\b(he|how|i|it|she|that|there|these|they|we|what|where|which|who|you)'?ll\b", r'\g<1> will', text,
                  flags=flags)
    text = re.sub(
        r"\b(everybody|everyone|he|how|it|she|somebody|someone|something|that|there|this|what|when|where|which|who|why)'?s\b",
        r'\g<1> is', text, flags=flags)
    text = re.sub(r"\b(I)'?m'a\b", r'\g<1> am about to', text, flags=flags)
    text = re.sub(r"\b(I)'?m'o\b", r'\g<1> am going to', text, flags=flags)
    text = re.sub(r"\b(I)'?m\b", r'\g<1> am', text, flags=flags)
    text = re.sub(r"\bshan't\b", 'shall not', text, flags=flags)
    text = re.sub(
        r"\b(are|could|did|does|do|go|had|has|have|is|may|might|must|need|ought|shall|should|was|were|would)n'?t\b",
        r'\g<1> not', text, flags=flags)
    text = re.sub(
        r"\b(could|had|he|i|may|might|must|should|these|they|those|to|we|what|where|which|who|would|you)'?ve\b",
        r'\g<1> have', text, flags=flags)
    text = re.sub(r"\b(how|so|that|there|these|they|those|we|what|where|which|who|why|you)'?re\b", r'\g<1> are', text,
                  flags=flags)
    text = re.sub(r"\b(I|it|she|that|there|they|we|which|you)'?d\b", r'\g<1> had', text, flags=flags)
    text = re.sub(r"\b(how|what|where|who|why)'?d\b", r'\g<1> did', text, flags=flags)

    return text


class ExpandContractionsComponent:
    name = "expand_contractions"

    def __init__(self, nlp: Language):
        self.nlp = nlp

    def __call__(self, doc: Doc) -> Doc:
        text = doc.text
        text = expand_contractions(text)
        return self.nlp.make_doc(text)


@Language.factory('expand_contractions')
def create_expand_contractions(nlp, name):
    return ExpandContractionsComponent(nlp)


def clean_docs(texts, rejoin=False):
    def clean_text(text):
        text = re.sub(r'@[A-Za-z0-9_-]{1,15}\b', " ", text)
        text = re.sub(r'https?://[A-Za-z0-9./]+', " ", text)
        text = re.sub(r'&amp;|&quot;', " ", text)
        return text

    texts = [clean_text(text) for text in texts]

    docs = nlp.pipe(texts, disable=['parser', 'ner', 'textcat', 'tok2vec'], batch_size=10_000)
    if 'expand_contractions' not in nlp.pipe_names:
        nlp.add_pipe('expand_contractions', before='tagger')

    docs_cleaned = []
    for doc in docs:
        doc = [token for token in doc if token.is_alpha]
        tokens = [tok.lemma_.strip() for tok in doc]

        if rejoin:
            tokens = ' '.join(tokens)
        docs_cleaned.append(tokens)

    return docs_cleaned


class SpacyTextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, rejoin=False):
        self.rejoin = rejoin

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return clean_docs(X, self.rejoin)

@app.route('/')
def index():
    print('Request for index page received')
    return render_template('index.html')

@app.errorhandler(400)
def bad_request_error(error):
    return jsonify({'error': 'Bad Request'}), 400

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not Found'}), 404

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form.get('tweet')

    if tweet:
        text_cleaned = clean_docs([tweet])
        padded_sequences = prepare_keras_data(text_cleaned)
        padded_sequences = padded_sequences.astype(np.float32)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], padded_sequences)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        sentiment_score = prediction[0][0]
        sentiment_class = "Positive" if sentiment_score > 0.5 else "Negative"

        return jsonify(sentiment_class=sentiment_class, sentiment_score=sentiment_score)
    else:
        return jsonify(error="No tweet provided"), 400

@app.route('/predict_page', methods=['POST'])
def predict_page():
    tweet = request.form.get('tweet')

    if tweet:
        # Appel à la fonction predict
        response = predict()
        if response.status_code == 200:
            prediction = response.get_json()
            sentiment_class = prediction["sentiment_class"]
            sentiment_score = prediction["sentiment_score"]
            return render_template('prediction.html', sentiment_class=sentiment_class, sentiment_score=sentiment_score)
        else:
            # Gérer l'erreur de /predict
            return render_template('error.html', error=response.get_json()["error"])
    else:
        return redirect(url_for('index'))




if __name__ == '__main__':
    setup()
    app.run(debug=True)
