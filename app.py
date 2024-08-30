import streamlit as st
import numpy as np
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing tools
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

#Load trained model
model = load_model('rnn.h5')

# Load the tokenizer for later use
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Preprocess function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

# Function to predict sentiment
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    sequenced_text = tokenizer.texts_to_sequences([processed_text])
    
    max_len = 100
    padded_text = pad_sequences(sequenced_text, maxlen=max_len, padding='post')
    
    word_count = len(text)

    if word_count < 5:
        return "Input must contain at least 5 words"

    prediction = model.predict(padded_text)
    labels = ['positive','neutral','negative']

    if prediction.shape[1] != len(labels):
        return "The input text could not be processed. Please try again."
    
    return labels[np.argmax(padded_text)]

# Streamlit app
st.title('Financial News Sentiment Analysis App')

# User input
user_input = st.text_area('Enter your financial news statement:')

if st.button('Predict Sentiment'):
    if user_input:
        word_count = len(word_tokenize(user_input))
        if word_count < 5:
            st.write("Input must contain at least 5 words")
        else:
            sentiment = predict_sentiment(user_input)
            st.write(f'The predicted sentiment is: {sentiment}')
    else:
        st.write('Please enter some text to predict.')
