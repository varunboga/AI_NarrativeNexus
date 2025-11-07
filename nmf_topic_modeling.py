import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from text_processing import preprocess_series
from topic_utils import display_topics
import joblib
import os

df = pd.read_csv("C:/Users/Varun Boga/OneDrive/Desktop/AI Narrative Nexus/data/req_data/20news-18828/20news_18828_clean.csv")
texts = preprocess_series(df["text"])

tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_tfidf = tfidf_vectorizer.fit_transform(texts)

nmf = NMF(n_components=20, random_state=42, max_iter=200)
nmf.fit(X_tfidf)

print("\n🔹 NMF Topics:\n")
display_topics(nmf, tfidf_vectorizer.get_feature_names_out())

os.makedirs("C:/Users/Varun Boga/OneDrive/Desktop/AI Narrative Nexus/model/Topic_modeling", exist_ok=True)
joblib.dump(nmf, "C:/Users/Varun Boga/OneDrive/Desktop/AI Narrative Nexus/model/Topic_modeling/nmf_model.pkl")
joblib.dump(tfidf_vectorizer, "C:/Users/Varun Boga/OneDrive/Desktop/AI Narrative Nexus/model/Topic_modeling/nmf_vectorizer.pkl")
print("\n NMF model and vectorizer saved.")
