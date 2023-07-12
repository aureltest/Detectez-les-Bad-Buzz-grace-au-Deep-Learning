from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import re
import spacy
from spacy.tokens import Doc
from spacy.language import Language
import nltk
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np

app = Flask(__name__)

# Charger le tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Charger le modèle
model = load_model('LSTM_model.h5')

#Charger spacy
nlp = spacy.load('en_core_web_lg')

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


def clean_docs(texts, lemmatize=False, stem=False, rejoin=False):
    if lemmatize and stem:
        raise ValueError("Un seul transformateur peut être appliqué.")

    def clean_text(text):
        text = re.sub(r'@[A-Za-z0-9_-]{1,15}\b', " ", text)
        text = re.sub(r'https?://[A-Za-z0-9./]+', " ", text)
        text = re.sub(r'&amp;|&quot;', " ", text)
        return text

    texts = [clean_text(text) for text in texts]

    docs = nlp.pipe(texts, disable=['parser', 'ner', 'textcat', 'tok2vec'], batch_size=10_000)
    if 'expand_contractions' not in nlp.pipe_names:
        nlp.add_pipe('expand_contractions', before='tagger')
    stemmer = nltk.PorterStemmer()

    docs_cleaned = []
    for doc in docs:
        doc = [token for token in doc if token.is_alpha]
        if lemmatize:
            tokens = [tok.lemma_.strip() for tok in doc]
        elif stem:
            tokens = [stemmer.stem(tok.text.strip()) for tok in doc]
        else:
            tokens = [tok.text.strip() for tok in doc]

        if rejoin:
            tokens = ' '.join(tokens)
        docs_cleaned.append(tokens)

    return docs_cleaned


class SpacyTextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, lemmatize=False, stem=False, rejoin=False):
        self.lemmatize = lemmatize
        self.stem = stem
        self.rejoin = rejoin

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return clean_docs(X, self.lemmatize, self.stem, self.rejoin)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    text_cleaned = clean_docs([data["message"]], lemmatize=True)
    padded_sequences = prepare_keras_data(text_cleaned)

    prediction = model.predict(np.array(padded_sequences))
    sentiment_score = prediction[0][0]
    sentiment_class = "positive" if sentiment_score > 0.5 else "negative"


    return jsonify({'sentiment_score': float(sentiment_score), 'sentiment_class': sentiment_class})


if __name__ == '__main__':
    app.run(debug=True)