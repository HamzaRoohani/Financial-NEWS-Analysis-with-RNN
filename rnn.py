import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, Flatten

# Downloading dataset of stopwords
nltk.download('punkt_tab')
nltk.download('stopwords')

# Loading dataset
df = pd.read_csv('Test.csv')
print(df)

# NLTK tools
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

# Tokenization of data
tokenize = Tokenizer()
tokenize.fit_on_texts(df['text'])
sequences = tokenize.texts_to_sequences(df['text'])
max_len = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_len, padding='post')
y = df['label']

# Model creation
model = Sequential()
model.add(Embedding(input_dim=len(tokenize.word_index)+1, output_dim=100, input_length=max_len))
model.add(SimpleRNN(64, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

fit = model.fit(X, y, epochs=5)

model.save('sentiment_analysis_with_rnn.h5')