from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the model
model = joblib.load('model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Extract features from the input JSON data
        features = np.array([[
            data['longitude'],
            data['latitude'],
            data['housing_median_age'],
            data['total_rooms'],
            data['total_bedrooms'],
            data['population'],
            data['households'],
            data['median_income']
        ]])
        # Make a prediction using the trained model
        prediction = model.predict(features)
        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0]})
    except KeyError as e:
        return jsonify({'error': f'Missing key: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(debug=True)
