from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        fever = int(request.form['fever'])
        chest_pain = int(request.form['chest_pain'])
        breathing = int(request.form['breathing'])

        input_data = np.array([[age, fever, chest_pain, breathing]])
        prediction = model.predict(input_data)[0]

        urgency_map = {0: "Low", 1: "Medium", 2: "High"}
        result = urgency_map.get(prediction, "Unknown")

        return render_template('result.html', result=result)
    
    except:
        return "Invalid input. Please try again."

if __name__ == '__main__':
    app.run(debug=True)
