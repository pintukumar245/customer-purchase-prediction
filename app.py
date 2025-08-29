from flask import Flask, render_template, request
import numpy as np
import joblib
import os

# Flask app
app = Flask(__name__)

# Load trained model (model.pkl apna trained model ka naam hoga)
model_path = os.path.join(os.path.dirname(__file__), "scaler.joblib")
model = joblib.load(model_path)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # User inputs
        user_id = request.form.get("user_id")  # Optional, prediction me use nahi hoga
        gender = request.form.get("gender")
        age = request.form.get("age")
        salary = request.form.get("salary")

        # Convert Gender to numeric
        if gender.lower() == "male":
            gender = 1
        else:
            gender = 0

        # Convert Age and Salary to numeric
        age = int(age)
        salary = float(salary)

        # Make feature array
        features = np.array([[int(user_id),gender, age, salary]])

        # Prediction
        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "Customer will Purchase ✅"
        else:
            result = "Customer will NOT Purchase ❌"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)
