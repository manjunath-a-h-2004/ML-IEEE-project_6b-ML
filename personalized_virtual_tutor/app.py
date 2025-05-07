from flask import Flask, render_template, request
import random

# Dummy recommender logic (replace this with your actual ANN/Bandit logic)
class DummyTutor:
    def recommend(self, topic, score):
        if score < 50:
            return f"Revise {topic} again for better understanding."
        else:
            return f"Proceed to next topic related to {topic}."

# Initialize
tutor = DummyTutor()
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        topic = request.form['topic']
        previous_score = float(request.form['score'])

        recommendation = tutor.recommend(topic, previous_score)

        return render_template('result.html', topic=topic, score=previous_score, recommendation=recommendation)
    except Exception as e:
        return f"Error: {str(e)}"

# Run app
if __name__ == '__main__':
    app.run(debug=True)
