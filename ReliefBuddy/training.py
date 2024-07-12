import sys
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
import os
import re

# Set default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Ensure you have the necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load intents JSON file
intents_file = os.path.join(current_dir, 'intents.json')
with open(intents_file, encoding='utf-8') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Function to handle contractions
def replace_contractions(text):
    contractions_dict = {
        "i'm": "i am",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "we're": "we are",
        "they're": "they are",
        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",
        "i'd": "i would",
        "you'd": "you would",
        "he'd": "he would",
        "she'd": "she would",
        "we'd": "we would",
        "they'd": "they would",
        "i'll": "i will",
        "you'll": "you will",
        "he'll": "he will",
        "she'll": "she will",
        "we'll": "we will",
        "they'll": "they will",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "doesn't": "does not",
        "don't": "do not",
        "didn't": "did not",
        "won't": "will not",
        "wouldn't": "would not",
        "shan't": "shall not",
        "shouldn't": "should not",
        "can't": "cannot",
        "couldn't": "could not",
        "mustn't": "must not",
        "let's": "let us",
        "that's": "that is",
        "who's": "who is",
        "what's": "what is",
        "here's": "here is",
        "there's": "there is",
        "where's": "where is",
        "when's": "when is",
        "why's": "why is",
        "how's": "how is"
    }
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text.lower())

# Preprocess the data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern = replace_contractions(pattern)
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

# Save words and classes for future use
pickle.dump(words, open(os.path.join(current_dir, 'words.pkl'), 'wb'))
pickle.dump(classes, open(os.path.join(current_dir, 'classes.pkl'), 'wb'))

training = []
output_empty = [0] * len(classes)

# Create training data
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training)

train_x = training[:, :len(words)]
train_y = training[:, len(words):]

# Split data into training and validation sets
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(train_x[0]),)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Define early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(train_x, train_y, epochs=200, batch_size=5, validation_data=(val_x, val_y), callbacks=[early_stopping], verbose=1)
model.save(os.path.join(current_dir, 'chatbot_model.keras'))  # Save the model in Keras format

print('Done')
