import pandas as pd
import pickle
import os
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# ✅ Load only the columns we need
df = pd.read_csv("spam.csv", encoding='latin-1')[['class', 'message']]
df.columns = ['label', 'message']  # Rename for consistency

# ✅ Map 'spam' to 1 and 'ham' to 0
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ✅ Prepare data
X = df['message']
y = df['label']

# ✅ Convert text to vectors
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# ✅ Train model
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# ✅ Save model and vectorizer
os.makedirs('models', exist_ok=True)
with open('models/spam_classifier_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved in 'models/' folder.")
# Removed invalid line
import pickle
import numpy as np
import re
app = Flask(__name__)
app = Flask(__name__)

# Load model & vectorizer
model = pickle.load(open('models/spam_classifier_model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message')
    if not message:
        return jsonify(result='error')

    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]
    label = 'spam' if prediction == 1 else 'ham'
    return jsonify(result=label)

if __name__ == '__main__':
    app.run(debug=True)
