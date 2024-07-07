import random
import json
import pickle
import numpy as np
import nltk
import sys
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import streamlit as st

@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    return True

# Download NLTK data if not already present
download_nltk_data()

sys.stdout.reconfigure(encoding='utf-8')

lemmatizer = WordNetLemmatizer()
model = load_model('/workspaces/ReliefBuddy-AI_Mental_Support_Chatbot/ReliefBuddy/chatbot_model.h5')

intents = json.loads(open('/workspaces/ReliefBuddy-AI_Mental_Support_Chatbot/ReliefBuddy/intents.json', encoding='utf-8').read())
words = pickle.load(open('/workspaces/ReliefBuddy-AI_Mental_Support_Chatbot/ReliefBuddy/words.pkl', 'rb'))
classes = pickle.load(open('/workspaces/ReliefBuddy-AI_Mental_Support_Chatbot/ReliefBuddy/classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, that's not clear to me."
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def main():
    st.title("ReliefBuddy - AI Mental Support Chatbot")
    st.write("Welcome to ReliefBuddy. How can I help you today?")

    user_input = st.text_input("You: ", "")

    if user_input:
        predicted_intents = predict_class(user_input)
        response = get_response(predicted_intents, intents)
        st.text_area("ReliefBuddy:", response, height=200)

if __name__ == "__main__":
    main()
