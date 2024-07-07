import sys
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Set default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Ensure you have the necessary NLTK data
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

nltk.data.path.append('D:/nltk_data')

lemmatizer = WordNetLemmatizer()

# Load intents JSON file
intents = json.loads(open('intents.json', encoding='utf-8').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Preprocess the data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

# Save words and classes for future use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

# Create training data
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Split data into training and validation sets
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

# Define and compile the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(len(trainX[0]),)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Try using Adam optimizer
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Define early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(trainX, trainY, epochs=200, batch_size=5, validation_data=(valX, valY), callbacks=[early_stopping], verbose=1)
model.save('chatbot_model.h5')

print('Done')
