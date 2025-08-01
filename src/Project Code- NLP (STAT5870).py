"""### Data Preprocessing"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

bbc_news=pd.read_csv("bbc-text.csv")

"""### Text Preprocessing"""

import string
import spacy
import nltk
import re

nltk.download("stopwords")
from nltk.corpus import stopwords

nlp=spacy.load("en_core_web_lg")

class Text_Processing:
    def __init__(self,sentence,drop_stopword=True):
        self.sentence=sentence
        self.drop_stopword=drop_stopword
    
    def lower_case(self,text):
        return text.lower()
    
    def stop_words(self,text):
        if self.drop_stopword == True:
            stop_words=stopwords.words("english")
            words=[word for word in text.split(" ") if word not in stop_words]
            
            return " ".join(words)
        else:
            return text
    
    def lemmatiation(self,text):
        doc=nlp(text)
        words=[word.lemma_ for word in doc]
        return " ".join(words)
        
    def processing_text(self):
        # Remove Punctations
        text=re.sub("[^A-Za-z]"," ",self.sentence)
        
        # Change to lower case
        text=self.lower_case(text)
        return text

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

bbc_news["clean_text"]=bbc_news["text"].apply(lambda text:Text_Processing(text).processing_text())
bbc_news["clean_text"]=bbc_news["clean_text"].apply(lambda text:Text_Processing(text).stop_words(text))
bbc_news["clean_text"]=bbc_news["clean_text"].apply(lambda text:Text_Processing(text).lemmatiation(text))
bbc_news["label"]=le.fit_transform(bbc_news["category"])

"""## Vectorization"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

X_train,X_test,y_train,y_test= train_test_split(bbc_news["clean_text"],bbc_news["label"],test_size=0.2,random_state=44)
count_vect = CountVectorizer()
Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(bbc_news["clean_text"])
count_vect.fit(bbc_news["clean_text"])
X_train_TFIDF = Tfidf_vect.transform(X_train)
X_test_TFIDF = Tfidf_vect.transform(X_test)
X_train_CV = count_vect.transform(X_train)
X_test_CV = count_vect.transform(X_test)

"""## Build ML Models"""

# fit the training dataset on the NB classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report


NBModel = MultinomialNB()
SVMModel = svm.SVC(kernel='linear')
DTModel= DecisionTreeClassifier()

from sklearn.model_selection import cross_val_score


scores_acc = cross_val_score(NBModel, X_train_TFIDF, y_train, cv=5, scoring='accuracy')
scores_prec = cross_val_score(NBModel, X_train_TFIDF, y_train, cv=5, scoring='precision_weighted')
scores_rcl = cross_val_score(NBModel, X_train_TFIDF, y_train, cv=5, scoring='recall_weighted')

print("Accuracy: ",scores_acc.mean())
print("Precision: ", scores_prec.mean())
print("Recall: ",scores_rcl.mean())

from sklearn.model_selection import cross_val_score


scores_acc = cross_val_score(SVMModel, X_train_TFIDF, y_train, cv=5, scoring='accuracy')
scores_prec = cross_val_score(SVMModel, X_train_TFIDF, y_train, cv=5, scoring='precision_weighted')
scores_rcl = cross_val_score(SVMModel, X_train_TFIDF, y_train, cv=5, scoring='recall_weighted')

print("Accuracy: ",scores_acc.mean())
print("Precision: ", scores_prec.mean())
print("Recall: ",scores_rcl.mean())

"""## Evaluation and Choice of ML Model using Cross Validation"""

from sklearn.model_selection import cross_val_score


scores_acc = cross_val_score(DTModel, X_train_TFIDF, y_train, cv=5, scoring='accuracy')
scores_prec = cross_val_score(DTModel, X_train_TFIDF, y_train, cv=5, scoring='precision_weighted')
scores_rcl = cross_val_score(DTModel, X_train_TFIDF, y_train, cv=5, scoring='recall_weighted')

print("Accuracy: ",scores_acc.mean())
print("Precision: ", scores_prec.mean())
print("Recall: ",scores_rcl.mean())

"""## SVM Classification Evaluation"""

SVMModel.fit(X_train_TFIDF,y_train)
predictions_SVM = SVMModel.predict(X_test_TFIDF)

print("SVM Classification using TFIDF Vectorizer Report  : ")
print(classification_report(predictions_SVM, y_test))
print("Overall Accuracy Score : ",accuracy_score(predictions_SVM, y_test)*100)

"""## Extras"""

bbc_news.head(n=5)

sns.countplot(x = "category", data = bbc_news)

bbc_news["clean_text"]=bbc_news["text"].apply(lambda text:Text_Processing(text).processing_text())
bbc_news.head(n=5)

bbc_news["clean_text"]=bbc_news["clean_text"].apply(lambda text:Text_Processing(text).stop_words(text))
bbc_news.head(n=5)

bbc_news["clean_text"]=bbc_news["clean_text"].apply(lambda text:Text_Processing(text).lemmatiation(text))
bbc_news.head(n=5)