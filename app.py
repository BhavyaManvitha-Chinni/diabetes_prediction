import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the scaler and model
scaler = pickle.load(open('sc.pkl', 'rb'))
model = pickle.load(open('classifier.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from form and convert to float
        float_features = [float(x) for x in request.form.values()]
        final_features = np.array([float_features])

        # Scale features and make prediction
        scaled_features = scaler.transform(final_features)
        prediction = model.predict(scaled_features)
        
        # Assuming the model returns a single prediction
        output = prediction[0]

        return render_template('result.html', prediction=output)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
