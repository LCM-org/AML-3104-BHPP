# Flask Web Application

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open('best_model.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        size = float(request.form['size'])
        total_sqft = float(request.form['total_sqft'])
        bath = float(request.form['bath'])
        balcony = float(request.form['balcony'])

        # Make prediction using the model
        features = np.array([[size, total_sqft, bath, balcony]])
        prediction = best_model.predict(features)[0]

        # Round the prediction to 2 decimal places
        prediction = round(prediction, 2)

        return render_template('result.html', prediction=prediction)
    except ValueError as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
