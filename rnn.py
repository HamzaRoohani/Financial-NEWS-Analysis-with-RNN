import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Downloading dataset of stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Loading dataset
df = pd.read_csv('sentiment.csv')

# NLTK tools
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

# Tokenization of data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'].values)
X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X, padding='post')
y = pd.get_dummies(df['label']) # Using One-Hot Encoding
print('Labels:',y)
print('Found unique tokens', len(X))

# Save the tokenizer after training
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model creation
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100))
model.add(LSTM(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

fit = model.fit(X, y, epochs=10)
model.save('rnn.h5')

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Model Prediction
pred_probs = model.predict(X_test)
print(pred_probs)

var = ["Growth is expected to continue in 2008 ."]
var_sequence = tokenizer.texts_to_sequences(var)
padded_var = pad_sequences(var_sequence, maxlen=X.shape[1], padding='post')
print(padded_var)
pred = model.predict(padded_var)
labels = ['positive','neutral','negative']
print(pred, labels[np.argmax(pred)])
