import nltk
import os

# Set the path for nltk data
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")

if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Add the nltk data directory to nltk's data path
nltk.data.path.append(nltk_data_dir)
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

print("Punkt tokenizer downloaded successfully.")