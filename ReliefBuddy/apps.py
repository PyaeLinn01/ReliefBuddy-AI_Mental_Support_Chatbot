from flask import Flask, request, jsonify, render_template
import random
import json
import pickle
import numpy as np
import nltk
import sys
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import os

sys.stdout.reconfigure(encoding='utf-8')
app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
current_dir = os.path.dirname(os.path.abspath(__file__))

model = load_model(os.path.join(current_dir, 'chatbot_model.h5'))
with open(os.path.join(current_dir, 'intents.json'), encoding='utf-8') as file:
    intents = json.load(file)
words = pickle.load(open(os.path.join(current_dir, 'words.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(current_dir, 'classes.pkl'), 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response_route():
    data = request.get_json()
    message = data['message']
    ints = predict_class(message)
    res = get_response(ints, intents)
    return jsonify({'response': res})

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000, debug=True)

