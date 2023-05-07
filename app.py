import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('XGBOOST.pkl', 'rb'))
scaler = pickle.load(open('Scaler1.pkl', 'rb'))

# Define the mapping of numerical labels to text labels
label_map = {
    0: 'Good',
    1: 'Poor',
    2: 'Standard'
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Get the input data from the request
    data = request.json['data']
    
    # Scale the input data using the scaler
    scaled_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    
    # Get the predicted label from the model
    prediction = model.predict(scaled_data)[0]
    
    # Map the predicted label to the corresponding text label
    prediction_text = label_map[prediction]
    
    # Return the predicted label as a JSON response
    return jsonify({'prediction': prediction_text})

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    data = [float(x) for x in request.form.values()]
    
    # Scale the input data using the scaler
    scaled_data = scaler.transform(np.array(data).reshape(1,-1))
    
    # Get the predicted label from the model
    prediction = model.predict(scaled_data)[0]
    
    # Map the predicted label to the corresponding text label
    prediction_text = label_map[prediction]
    
    # Return the predicted label as the output of the template
    return render_template("home.html", prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
