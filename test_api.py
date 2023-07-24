import unittest
import json
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

    def test_prediction_page(self):
        """
        Teste si la page de prédiction est accessible et réactive
        """
        response = self.app.post('/predict', data={'tweet': 'This is a test tweet.'})
        self.assertEqual(response.status_code, 200)

    def test_prediction_page_returns_prediction(self):
        """
        Teste si la page de prédiction renvoie une prédiction
        """
        response = self.app.post('/predict', data={'tweet': 'This is a test tweet.'})
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('prediction', data)

    def test_prediction_page_handles_errors(self):
        """
        Teste si la page de prédiction gère correctement les erreurs
        """
        response = self.app.post('/predict', data={})
        self.assertEqual(response.status_code, 400)

    def test_non_existent_route(self):
        """
        Teste une route qui n'existe pas pour vérifier le bon code d'état
        """
        response = self.app.get('/non_existent_route', content_type='html/text')
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
