from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your machine learning model
with open('models/cv.pkl', 'rb') as cv_file, open('models/clf.pkl', 'rb') as clf_file:
    cv = pickle.load(cv_file)
    clf = pickle.load(clf_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        email = data.get('email', '')
        tokenized_email = cv.transform([email])
        prediction = clf.predict(tokenized_email)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)