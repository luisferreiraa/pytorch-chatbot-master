# AI Chatbot with Flask and PyTorch  
This project is a simple AI-powered chatbot built using Flask (for the web interface) and PyTorch (for training and running the deep learning model). The chatbot processes natural language queries and returns appropriate responses based on the trained model. It's designed to handle various intents and can be expanded to include more functionality and intens over time.

## Table of Contents

- Technologies Used
- Project Structure
- Setup Instructions
- How it Works
- Usage
- License

## Technologies Used

- Flask: A lightweight WSGI web application framework in Python.
- Pytorch: An open-source deep learning framework for building and training the neural network.
- NLTK (Natural Language Toolkit): A suite of libraries for building and training the neural network.
- HTML/CSS/JavaScript: Used for rendering the web interface for user interaction.

## Project Structure

```console
├── model.py               # Defines the structure of the neural network.
├── nltk_utils.py          # Utility functions for processing text (tokenization, bag of words, etc.).
├── data.pth               # Pre-trained model data file.
├── intents.json           # List of intents, patterns, and responses used to train the model.
├── app.py                 # Flask application that handles the chatbot logic and endpoints.
├── static/                # Contains static files such as the CV file.
│   └── cv_luis.pdf        # The CV to be downloaded by the user.
└── templates/
    └── home.html          # Main HTML file for rendering the web interface.
```

## Setup Instructions

### Prerequisites

- Python 3.x
- PyTorch
- Flask
- NLTK

### Installation

1. Clone the repository:

```console
git clone https://github.com/luisferreiraa/pytorch-cv-chatbot.git
cd pytorch-cv-chatbot
```

2. Set up a virtual environment (optional but recommended):

```console
python -m venv venv
source venv/bin/activate # On Windows use: venv\Scripts\activate
```

3. Install dependencies:

```console
pip install -r requirements.txt
```

4. Download NLTK resources: Inside a Python shell, run the following commands to download required resources:

```console
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

5. Run the Flask application:

```console
python app.py
```

The application should be running on http://localhost:5000.

## How it Works

1. Natural Language Processing: The user's input is tokenized and converted into a "bag of words" representation using the nltk_utils.py module. The pre-trained model processes this input and predicts which intent (or category) the user's query belongs to.
2. Neural Network: The chatbot uses a simple feed-forward neural network (NeuralNet class in model.py). This model was trained using supervised learning on the data found in intents.json. The model classifies the input into one of the predefined categories (or tags) and responds accordingly.
3. Handling Responses: After determining the intent, the application selects a random response from the list associated with that intent in the intents.json file.
4. CV Download: The application has an endpoint (/cv) that allows the user to download a CV in PDF format when requested via a specific query (e.g., "Send me your CV").

## Usage

1. Web Interface: Open http://localhost:5000 in your browser. You can interact with the chatbot by typing a question or command. It will respond based on its training data.
2. Example Queries:

- "Send me your CV"
- "What can you do?"
- "Tell me a joke"

3. Adding new intents: This chatbot was trained to act as my CV personal assistant. It knows everything about my studies, experience and focus. It answers in pt-PT. You can extend the chatbot by editing the intens.json file. Add new patterns, responses, and tags to teach the bot new capabilities. After adding, make sure to retrain your model.

## License

This project is open-source and available under the MIT License.
