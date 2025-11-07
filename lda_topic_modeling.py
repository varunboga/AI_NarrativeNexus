import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from text_processing import preprocess_series
from topic_utils import display_topics
import joblib
import os

df = pd.read_csv("C:/Users/Varun Boga/OneDrive/Desktop/AI Narrative Nexus/data/req_data/20news-18828/20news_18828_clean.csv")
texts = preprocess_series(df["text"])

count_vectorizer = CountVectorizer(max_features=10000, stop_words="english")
X_counts = count_vectorizer.fit_transform(texts)

lda = LatentDirichletAllocation(
    n_components=20, 
    random_state=42,
    learning_method="batch",
    max_iter=10
)
lda.fit(X_counts)

print("\n🔹 LDA Topics:\n")
display_topics(lda, count_vectorizer.get_feature_names_out())

os.makedirs("C:/Users/Varun Boga/OneDrive/Desktop/AI Narrative Nexus/model/Topic_modeling", exist_ok=True)
joblib.dump(lda, "C:/Users/Varun Boga/OneDrive/Desktop/AI Narrative Nexus/model/Topic_modeling/lda_model.pkl")
joblib.dump(count_vectorizer, "C:/Users/Varun Boga/OneDrive/Desktop/AI Narrative Nexus/model/Topic_modeling/lda_vectorizer.pkl")
print("\n LDA model and vectorizer saved.")
