import unittest
import json
import numpy as np
from spacy.tokens import Doc
from flask import Flask
import Testelin_Aurelien_1_modele_072023 as main


class FlaskTest(unittest.TestCase):
    """
    Classe de tests pour l'API Flask
    """

    def setUp(self):
        """
        Configuration initiale avant chaque test
        """
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.app = main.app.test_client()

    def test_index_page(self):
        """
        Teste si la page d'accueil est accessible
        """
        response = self.app.get('/', content_type='html/text')
        self.assertEqual(response.status_code, 200)

    def test_api_predict_sentiment(self):
        """
        Teste la route API api_predict_sentiment pour vérifier que la prédiction est renvoyée correctement.
        Vérifie que la réponse est un JSON avec les champs sentiment_score et sentiment_class.
        """
        response = self.app.post('/api/predict_sentiment', json={'tweet': 'I love coding !'})
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)

        self.assertIn('sentiment_score', data)
        self.assertIn('sentiment_class', data)
        self.assertIsInstance(data['sentiment_score'], float)
        self.assertIn(data['sentiment_class'], ["Positive", "Negative"])

    def test_prediction_page(self):
        """
        Teste si la page de prédiction renvoie un code 200 si le tweet existe et un message d'erreur sinon
        """
        response = self.app.post('/predict_page', data={'tweet': 'I love coding !'})
        self.assertEqual(response.status_code, 200)

        response = self.app.post('/predict_page', data={'tweet': ''})
        self.assertEqual(response.status_code, 400)

    def test_prepare_keras_data(self):
        """
        Teste la fonction prepare_keras_data pour vérifier que les données sont correctement préparées pour l'entraînement.
        Vérifie que le résultat est un tableau numpy et que toutes les séquences sont de longueur 40.
        """
        input_data = ['This is a test', 'Another test']
        output_data = main.prepare_keras_data(input_data)

        assert isinstance(output_data, np.ndarray), "Output should be a numpy array."
        for sequence in output_data:
            assert len(sequence) == 40, "All sequences should be of length 40."

    def test_clean_docs(self):
        """
        Teste la fonction clean_docs pour vérifier que le texte est correctement nettoyé.
        Vérifie que les mentions, les liens et les caractères spéciaux sont correctement retirés du texte.
        """
        texts = ["@John I love https://www.coding.com &amp; &quot;"]
        result = main.clean_docs(texts, rejoin=True) # corrected here
        expected_result = ['i love']
        self.assertEqual(result, expected_result)

    def test_expand_contractions(self):
        """
        Teste la fonction ExpandContractionsComponent pour vérifier que les contractions sont correctement étendues.
        Vérifie que les contractions comme "I'm" et "I'd've" sont étendues en "I am" et "I would have" respectivement.
        """
        component = main.ExpandContractionsComponent(main.nlp) # corrected here
        doc = Doc(main.nlp.vocab, words=["I'm", "gonna", "I'd've"])
        new_doc = component(doc)
        self.assertEqual(str(new_doc).strip(), "I am going to I would have")

    def test_predict_sentiment(self):
        """
        Teste la fonction predict_sentiment pour vérifier que le sentiment d'un tweet est correctement prédit.
        Vérifie que le score du sentiment est un float et que la classe du sentiment est soit "Positive" soit "Negative".
        """
        tweet = "I love coding"
        sentiment_score, sentiment_class = main.predict_sentiment(tweet)

        # Vérifiez que le score du sentiment est un float
        assert isinstance(sentiment_score, float), "Sentiment score should be a float."

        # Vérifiez que la classe du sentiment est soit "Positive" soit "Negative"
        assert sentiment_class in ["Positive", "Negative"], "Sentiment class should be either 'Positive' or 'Negative'."


if __name__ == "__main__":
    unittest.main()
