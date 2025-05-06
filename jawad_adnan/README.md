# ML_Project_6B

🏥 Medical Urgency Prediction System
A simple machine learning web application that predicts a patient's urgency level (Low, Medium, High) based on four inputs: age, fever, chest pain, and breathing difficulty.

📌 Project Overview
This project is built using:

Python

Scikit-learn

Pandas

Flask

Joblib

It is intended to assist medical personnel in triaging patients quickly based on common symptoms and age.

🚀 Features
Accepts user input for:

Age

Fever (Yes/No)

Chest Pain (Yes/No)

Breathing Difficulty (Yes/No)

Predicts one of the following:

Low urgency (0)

Medium urgency (1)

High urgency (2)

Easy-to-use web interface using Flask

Uses a trained RandomForestClassifier for prediction

🧠 Model
The model was trained on mock data with the following structure:

text
Copy code
Columns: ['age', 'fever', 'chest_pain', 'breathing_difficulty']
Target: urgency_level (0 = Low, 1 = Medium, 2 = High)
The training data consists of manually created examples representing different combinations of symptoms.

🛠️ How to Run
1. Clone the Repository
bash
Copy code
git clone https://github.com/your-username/medical-urgency-predictor.git
cd medical-urgency-predictor
2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
Or manually install:

bash
Copy code
pip install flask pandas scikit-learn joblib
3. Train the Model (Optional)
If you want to retrain:

python
Copy code
# Run this in Jupyter or script
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
joblib.dump(model, 'model.pkl')

4. Start the Flask App
bash
Copy code
python app.py
5. Open in Browser
Go to http://127.0.0.1:5000 in your browser.

⚠️ Disclaimer
This is a demo project using mock data. It is not intended for real medical diagnosis or treatment decisions.

📬 Contact
Developer: Mohammed Jawad Hussain
Feel free to connect on LinkedIn or GitHub!







