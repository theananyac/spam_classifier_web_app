import os
import pickle
import string
import logging
import traceback
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import pandas as pd
import io
import nltk
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from nltk import pos_tag, word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model paths
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'spam_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
os.makedirs(MODEL_DIR, exist_ok=True)

# Global model variables
model = None
vectorizer = None

def load_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

def init_model():
    global model, vectorizer
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(VECTORIZER_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
            logger.info("Model loaded successfully")
        else:
            # Initialize with dummy data if no model exists
            X_train = ["free money", "hello", "win prize", "normal message"]
            y_train = [1, 0, 1, 0]
            
            vectorizer = CountVectorizer()
            X_vec = vectorizer.fit_transform(X_train)
            
            model = MultinomialNB()
            model.fit(X_vec, y_train)
            
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(model, f)
            with open(VECTORIZER_PATH, 'wb') as f:
                pickle.dump(vectorizer, f)
            logger.info("Created new dummy model")
    except Exception as e:
        logger.error(f"Model init failed: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        if request.is_json:
            data = request.get_json()
            message = data.get('message', '')
        else:
            message = request.form.get('message', '')

        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Language detection and translation
        try:
            lang = detect(message)
            if lang != 'en':
                message = GoogleTranslator(source=lang, target='en').translate(message[:5000])
        except:
            lang = 'en'

        # Feature extraction
        features = {
            'length': len(message),
            'word_count': len(message.split()),
            'punct_count': sum(1 for c in message if c in string.punctuation)
        }

        # POS tagging
        try:
            tokens = word_tokenize(message)
            pos_tags = pos_tag(tokens)
            features['pos'] = {tag: 0 for _, tag in pos_tags}
            for _, tag in pos_tags:
                features['pos'][tag] += 1
        except:
            features['pos'] = {}

        # Prediction
        X = vectorizer.transform([message])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][pred]
        
        return jsonify({
            'result': 'Spam' if pred == 1 else 'Ham',
            'confidence': round(float(proba) * 100, 2),
            'language': lang,
            'features': features,
            'version': 'advanced'
        })

    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'CSV file required'}), 400

        df = pd.read_csv(file)
        if 'message' not in df.columns or 'label' not in df.columns:
            return jsonify({'error': "CSV needs 'message' and 'label' columns"}), 400

        # Train new model
        X_train, X_test, y_train, y_test = train_test_split(
            df['message'], df['label'], test_size=0.2, random_state=42
        )

        new_vectorizer = CountVectorizer()
        X_vec = new_vectorizer.fit_transform(X_train)

        new_model = MultinomialNB()
        new_model.fit(X_vec, y_train)

        # Save model
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(new_model, f)
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(new_vectorizer, f)

        global model, vectorizer
        model = new_model
        vectorizer = new_vectorizer

        return jsonify({'success': 'Model retrained successfully'})

    except Exception as e:
        logger.error(f"Training failed: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_nltk()
    init_model()
    app.run(host='0.0.0.0', port=5000, debug=True)