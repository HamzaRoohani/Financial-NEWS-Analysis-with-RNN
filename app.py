import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

# Preprocessing tools
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

#Load trained model
model = load_model('sentiment_analysis_with_rnn.h5')

tokenizer = Tokenizer()

# Preprocess function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

# Function to predict sentiment
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    sequenced_text = tokenizer.texts_to_sequences([processed_text])
    max_len = max(len(seq) for seq in sequenced_text)
    padded_text = pad_sequences(sequenced_text, maxlen=max_len ,padding='post')
    prediction = model.predict(padded_text)
    return 'Positive' if prediction[0][0]>0.5 else 'Negative'

# Streamlit app
st.title('Sentiment Analysis App')
st.write('Enter a movie review below')

# User input
user_input = st.text_area('Enter your review:')

if st.button('Predict Sentiment'):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f'The predicted sentiment is:{sentiment}')
    else:
        st.write('Please enter some text to predict.')
