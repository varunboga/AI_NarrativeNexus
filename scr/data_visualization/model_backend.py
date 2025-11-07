import joblib
import numpy as np
import pandas as pd

def load_topic_models():
    lda = joblib.load('model/Topic_modeling/lda_model.pkl')
    vectorizer = joblib.load('model/Topic_modeling/lda_vectorizer.pkl')
    return lda, vectorizer

def load_sentiment_models():
    rf = joblib.load('model/Sentiment_analysis/random_forest_model.pkl')
    vec = joblib.load('model/Sentiment_analysis/tfidf_vectorizer.pkl')
    return rf, vec

def predict_topics(texts, lda, vectorizer):
    X_counts = vectorizer.transform(texts)
    doc_topics = lda.transform(X_counts)
    assigned_topics = np.argmax(doc_topics, axis=1)
    return assigned_topics

def predict_sentiments(texts, rf, vec):
    X = vec.transform(texts)
    preds = rf.predict(X)
    return preds

def analyze_data(df):
    lda, lda_vec = load_topic_models()
    rf, rf_vec = load_sentiment_models()
    if "text" not in df.columns:
        raise ValueError("DataFrame must have a 'text' column")
    texts = df["text"].astype(str).tolist()
    topic_labels = predict_topics(texts, lda, lda_vec)
    sentiment_labels = predict_sentiments(texts, rf, rf_vec)
    df["assigned_topic"] = topic_labels
    df["sentiment_label"] = sentiment_labels
    return df
