import random
import json
import pickle
import numpy as np
import nltk
import sys
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
import streamlit as st
import os
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import pyttsx3

# Initialize Pyttsx3 for text-to-speech
engine = pyttsx3.init()

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Initialize NLTK lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load model and data
current_dir = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(current_dir, 'chatbot_model.keras'))

with open(os.path.join(current_dir, 'intents.json'), encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open(os.path.join(current_dir, 'words.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(current_dir, 'classes.pkl'), 'rb'))

# Load additional dataset
ds = load_dataset("Amod/mental_health_counseling_conversations")
contexts = ds['train']['Context']
responses = ds['train']['Response']

# Create a dictionary to map contexts to their respective responses
context_response_map = {}
for context, response in zip(contexts, responses):
    if context in context_response_map:
        context_response_map[context].append(response)
    else:
        context_response_map[context] = [response]

# Vectorize the contexts using TF-IDF
vectorizer = TfidfVectorizer().fit(contexts)
context_vectors = vectorizer.transform(contexts)

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Convert sentence to bag-of-words representation."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predict the intent of the given sentence."""
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json, message):
    """Get a response based on predicted intents or fallback to dataset response."""
    if not intents_list:
        return get_dataset_response(message)
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def get_dataset_response(message):
    """Get a response from the additional dataset based on cosine similarity."""
    message_vector = vectorizer.transform([message])
    similarities = cosine_similarity(message_vector, context_vectors)
    best_match_index = np.argmax(similarities)
    best_match_context = contexts[best_match_index]
    best_match_responses = context_response_map[best_match_context]
    return random.choice(best_match_responses)

def recognize_speech():
    """Recognize speech using the microphone and return as text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand audio")
        except sr.RequestError:
            st.error("Could not request results from Google Speech Recognition service")
    return ""

def speak_text(text):
    """Speak the text using text-to-speech."""
    engine.say(text)
    engine.runAndWait()

def main():
    """Streamlit interface for the chatbot."""
    st.title("ReliefBuddy - AI Mental Support Chatbot")
    st.write("Welcome to ReliefBuddy. How can I help you today?")

    if st.button("Use Microphone"):
        user_input = recognize_speech()
        st.write(f"You said: {user_input}")
    else:
        user_input = st.text_input("You: ", "")

    if user_input:
        predicted_intents = predict_class(user_input)
        response = get_response(predicted_intents, intents, user_input)
        st.text_area("ReliefBuddy:", response, height=200)
        speak_text(response)

if __name__ == "__main__":
    main()
