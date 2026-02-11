# 🔍 Détectez les Bad Buzz grâce au Deep Learning

> **Projet 7** — Parcours Ingénieur IA · OpenClassrooms

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![BERT](https://img.shields.io/badge/🤗_Transformers-BERT-yellow?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-0078D4?style=flat-square&logo=microsoftazure&logoColor=white)

---

## 🎯 Contexte du projet

En tant que consultant IA dans un cabinet de **marketing digital**, le client (une compagnie aérienne) souhaite anticiper les **bad buzz sur les réseaux sociaux**. L'objectif est de développer un modèle de Deep Learning capable de prédire le sentiment (positif / négatif) associé à un tweet, puis de le déployer en production dans le Cloud.

## 🧠 Compétences développées

- **Deep Learning pour le NLP** — Entraînement de modèles sur des données textuelles (LSTM, GRU, Transformers)
- **Word Embeddings** — Comparaison de techniques de plongement de mots (Word2Vec, GloVe, FastText, BERT)
- **MLOps** — Mise en place d'un pipeline d'entraînement reproductible et versionné
- **Déploiement Cloud** — Déploiement continu d'un moteur d'inférence dans le Cloud (Azure)
- **Rédaction technique** — Note méthodologique détaillant la démarche de modélisation

## 🔬 Approche méthodologique

```
1. Prétraitement des tweets (nettoyage, tokenisation, padding)
2. Approche classique — Régression Logistique + TF-IDF (baseline)
3. Approche Deep Learning — LSTM/GRU avec Word2Vec & GloVe
4. Approche Transformers — Fine-tuning de BERT/DistilBERT
5. Comparaison des performances (accuracy, F1-score, AUC)
6. Mise en place du pipeline MLOps (MLflow)
7. Déploiement de l'API d'inférence dans le Cloud
```

## 🛠️ Stack technique

| Catégorie | Technologies |
|-----------|-------------|
| **Langage** | Python 3.8+ |
| **Deep Learning** | TensorFlow/Keras, PyTorch |
| **NLP** | NLTK, Hugging Face Transformers, BERT, DistilBERT |
| **Embeddings** | Word2Vec, GloVe, FastText |
| **MLOps** | MLflow, Git/GitHub |
| **Cloud** | Microsoft Azure |
| **API** | Flask / FastAPI |
| **Environnement** | Jupyter Notebook, Google Colab (GPU) |

## 📁 Structure du projet

```
├── notebooks/
│   ├── 01_preprocessing.ipynb      # Nettoyage et exploration des tweets
│   ├── 02_modele_classique.ipynb   # Baseline (TF-IDF + Logistic Regression)
│   ├── 03_deep_learning.ipynb      # LSTM/GRU avec word embeddings
│   └── 04_transformers.ipynb       # Fine-tuning BERT
├── api/                            # API d'inférence
├── docs/
│   └── note_methodologique.pdf     # Note méthodologique
├── data/                           # Données (non incluses)
└── README.md
```

## 📌 Points clés

- Comparaison de **3 approches** : classique, Deep Learning, Transformers
- Progression significative des performances du baseline vers BERT
- Pipeline **MLOps complet** avec tracking d'expériences via MLflow
- **Déploiement continu** sur Azure avec API REST pour l'inférence en production
- Note méthodologique professionnelle documentant l'ensemble de la démarche

---

## 👤 Auteur

**Aurélien T.** — Data Scientist & Ingénieur IA  

---

*Projet réalisé dans le cadre du parcours [Ingénieur IA](https://openclassrooms.com/fr/paths/188-ingenieur-ia) d'OpenClassrooms*
