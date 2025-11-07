import joblib

def predict_rf(texts, model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    if isinstance(texts, str):
        texts = [texts]
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    return preds
