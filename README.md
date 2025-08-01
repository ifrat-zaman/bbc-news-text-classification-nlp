
# BBC News Article Text Classification using NLP

This project aims to classify BBC news articles into predefined genres using Natural Language Processing and Machine Learning techniques. It was developed as part of the "Big Data Analysis Using Python" course (STAT 5870) at Western Michigan University.

## 📌 Objective
To build a machine learning pipeline that preprocesses raw text data and classifies articles into five categories: **business, entertainment, politics, sport, and tech**.

## 📁 Dataset
- **Source:** [Kaggle - BBC Dataset](https://www.kaggle.com/datasets/sainijagjit/bbc-dataset)
- **Size:** 2225 articles with labeled genres

## 🔧 Methodology
1. **Preprocessing:**
   - Lowercasing
   - Stopword removal (NLTK)
   - Lemmatization (SpaCy)
   - Vectorization (CountVectorizer and TF-IDF)

2. **Models Trained:**
   - Multinomial Naive Bayes
   - Decision Tree
   - Support Vector Machine (SVM)

3. **Model Evaluation:**
   - Cross-validation (3, 5, 10 folds)
   - Metrics: Accuracy, Precision, Recall, F1-score

## 🏆 Best Result
- **Model:** SVM with linear kernel
- **Vectorizer:** TF-IDF
- **Accuracy:** 98.20%

## 📦 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK, SpaCy
- Matplotlib, Seaborn

## 📂 Folder Structure
```
bbc-news-text-classification-nlp/
├── data/
│   └── bbc-text.csv
├── notebooks/
│   └── project-code.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model_training.py
│   └── utils.py
├── outputs/
│   ├── figures/
│   └── models/
├── reports/
│   ├── Project Report.pdf
│   └── Project Report.docx
├── README.md
├── requirements.txt
└── .gitignore
```

## 📜 Authors
- Ifrat Zaman
- Gnana Deepak Madduri
- Asif Irfanullah Masum


## 🧾 License
This project is for academic and educational use only.
