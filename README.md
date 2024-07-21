# ReliefBuddy

ReliefBuddy is a chatbot application designed to provide support and assistance to individuals dealing with anxiety and other mental health challenges. The chatbot uses natural language processing (NLP) techniques and integrates with the OpenAI API to provide meaningful and empathetic responses.

## Features

- Responds to user queries about anxiety and provides helpful suggestions and support.
- Uses a trained machine learning model to identify user intents.
- Fetches responses from an external API when needed.
- Web-based interface using Flask for interaction.

## Installation

### Prerequisites

- Python 3.6 or later
- pip (Python package installer)
- Internet connection for API calls

### Steps

1. **Clone the Repository**

    ```bash
    git clone https://github.com/pyaelinn01/reliefbuddy.git
    cd reliefbuddy
    ```

2. **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download NLTK Data**

    Ensure you have the necessary NLTK data by running the following commands:

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    ```

5. **Set Up API Keys**

    Create a `.env` file in the root directory and add your OpenAI API key:

    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    ```

6. **Prepare Data**

    Make sure you have `intents.json`, `words.pkl`, and `classes.pkl` in the root directory. These files should contain the necessary data for training the model and for predictions.

## Usage

### Running the Application

1. **Start the Flask Server**

    ```bash
    python app.py
    ```

2. **Access the Application**

    Open your web browser and navigate to `http://localhost:5000` to interact with ReliefBuddy.

### API Endpoints

- **GET `/`**: Returns the home page.
- **POST `/get_response`**: Accepts a JSON payload with a user message and returns the chatbot's response.

    Example request:

    ```json
    {
        "message": "I'm feeling anxious"
    }
    ```

    Example response:

    ```json
    {
        "response": "Anxiety can be really tough. Have you tried grounding techniques, like focusing on your senses?"
    }
    ```

## Project Structure

- **app.py**: Main application file with Flask routes and logic.
- **training.py**: Script for training the machine learning model.
- **intents.json**: JSON file containing predefined intents and responses.
- **templates/**: Directory containing HTML templates for the web interface.
- **static/**: Directory containing static files like CSS and JavaScript.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to the developers of NLTK and TensorFlow for providing the tools needed for natural language processing and machine learning.
- Special thanks to OpenAI for their powerful language model API.

## Contact

For any questions or feedback, please contact [pltlosgot@gmail.com].

---
