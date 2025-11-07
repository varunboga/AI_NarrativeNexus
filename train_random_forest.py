import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

texts = ["I love this!", "Bad experience", "Absolutely great", "Horrible", "Will buy again", "Not recommended"]
labels = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(texts)
rf = RandomForestClassifier()
rf.fit(X, labels)

joblib.dump(rf, 'model/Sentiment_analysis/random_forest_model.pkl')
joblib.dump(tfidf, 'model/Sentiment_analysis/tfidf_vectorizer.pkl')
print("Saved RandomForest and TFIDF vectorizer.")
