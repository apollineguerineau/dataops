from flask import Flask, jsonify, request
import pandas as pd
app = Flask(__name__)


def encode_prenom(prenom: str) -> pd.Series:
    """
        This function encode a given name into a pd.Series.
        
        For instance alain is encoded [1, 0, 0, 0, 0 ... 1, 0 ...].
    """
    alphabet = "abcdefghijklmnopqrstuvwxyzé-'"
    prenom = prenom.lower()
    
    return pd.Series([letter in prenom for letter in alphabet]).astype(int)

# Placeholder for your gender prediction function
def predict_gender(name):
    # Assume a simple rule-based approach for demonstration purposes
    from joblib import load
    clf = load('model.bin') 
    encoded = encode_prenom(name)
    pred = clf.predict([encoded])[0]
    if pred == 0:
        gender = 'male'
    else:
        gender = 'female'
    
    return {'gender': gender}

@app.route('/predict/<name>', methods=['GET'])
def predict_gender_api(name):
    try:
        result = predict_gender(name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)