import numpy as np
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense  # type: ignore

texts = ["I love this!", "Bad experience", "Absolutely great", "Horrible", "Will buy again", "Not recommended"]
labels = np.array([1, 0, 1, 0, 1, 0])  # 1=positive, 0=negative

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=10)

model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=10),
    LSTM(8),
    Dense(2, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, labels, epochs=5, verbose=1)

model.save('model/Sentiment_analysis/lstm_model.h5')
joblib.dump(tokenizer, 'model/Sentiment_analysis/lstm_tokenizer.pkl')
print("Saved LSTM model and tokenizer.")
