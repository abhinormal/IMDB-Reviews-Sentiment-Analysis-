import streamlit as st
import os
import json

from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to load Kaggle credentials
def load_kaggle_credentials():
    kaggle_dictionary = json.load(open("kaggle.json"))
    os.environ["KAGGLE_USERNAME"] = kaggle_dictionary["username"]
    os.environ["KAGGLE_KEY"] = kaggle_dictionary["key"]

# Function to download and unzip dataset
def download_and_unzip_dataset():
    with ZipFile("IMDB Dataset.csv.zip", "r") as zip_ref:
        zip_ref.extractall()

# Function to load dataset
def load_dataset():
    data = pd.read_csv("IMDB Dataset.csv")
    return data

# Function to preprocess data
def preprocess_data(data):
    data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_data["review"])
    X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["review"]), maxlen=200)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen=200)
    Y_train = train_data["sentiment"]
    Y_test = test_data["sentiment"]
    return X_train, X_test, Y_train, Y_test, tokenizer

# Function to build LSTM model
def build_model(tokenizer):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

# Function to train the model
def train_model(model, X_train, Y_train):
    model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.2)

# Function to evaluate the model
def evaluate_model(model, X_test, Y_test):
    loss, accuracy = model.evaluate(X_test, Y_test)
    return loss, accuracy

# Function to predict sentiment
def predict_sentiment(model, tokenizer, review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment

# Load Kaggle credentials
load_kaggle_credentials()

# Download and unzip dataset
download_and_unzip_dataset()

# Load dataset
data = load_dataset()

# Preprocess data
X_train, X_test, Y_train, Y_test, tokenizer = preprocess_data(data)

# Build model
model = build_model(tokenizer)

# Train model
train_model(model, X_train, Y_train)

# Evaluate model
loss, accuracy = evaluate_model(model, X_test, Y_test)

# Streamlit app
st.title("Movie Review Sentiment Analysis")

# Sidebar for user input
review = st.text_area("Enter your movie review here:")

if st.button("Predict"):
    sentiment = predict_sentiment(model, tokenizer, review)
    st.write(f"The sentiment of the review is: {sentiment}")
