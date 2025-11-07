import numpy as np
import joblib
from tensorflow.keras.models import load_model # type: ignore

def predict_lstm(texts, model_path, tokenizer_path, maxlen=100):
    model = load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    if isinstance(texts, str):
        texts = [texts]
    seqs = tokenizer.texts_to_sequences(texts)
    seqs = np.array([s[:maxlen] for s in seqs])
    seqs = np.array([np.pad(s, (max(0, maxlen - len(s)), 0)) for s in seqs])
    preds = model.predict(seqs)
    y_pred = preds.argmax(axis=1)
    return y_pred
