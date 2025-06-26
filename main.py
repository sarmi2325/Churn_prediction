from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import os
from interpreter import interpret_prediction

app = Flask(__name__)

# Load XGBoost model
model = xgb.XGBClassifier()
model.load_model("model/churn_model.json")



# Define SHAP-selected features used in training
selected_features = [
    'InternetService_Fiber optic',
    'tenure',
    'is_multiple_services',
    'total_services',
    'charges_ratio',
    'TechSupport_No internet service',
    'MultipleLines_Yes',
    'StreamingTV_Yes',
    'StreamingMovies_Yes',
    'tenure_group_49-72'
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Get raw form input
    form = request.form
    tenure = int(form['tenure'])
    monthly = float(form['MonthlyCharges'])
    total = float(form['TotalCharges'])

    # 2. Feature engineering
    features = {
        'InternetService_Fiber optic': 1 if form['InternetService'] == 'Fiber optic' else 0,
        'tenure': tenure,
        'is_multiple_services': 1 if form['PhoneService'] == 'Yes' and form['InternetService'] != 'No' else 0,
        'total_services': int(form['total_services']),
        'charges_ratio': monthly / (total + 1),
        'TechSupport_No internet service': 1 if form['TechSupport'] == 'No internet service' else 0,
        'MultipleLines_Yes': 1 if form['MultipleLines'] == 'Yes' else 0,
        'StreamingTV_Yes': 1 if form['StreamingTV'] == 'Yes' else 0,
        'StreamingMovies_Yes': 1 if form['StreamingMovies'] == 'Yes' else 0,
        'tenure_group_49-72': 1 if 49 <= tenure <= 72 else 0
    }

    # 3. Create DataFrame
    input_df = pd.DataFrame([features])

    # 4. Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)
    shap_row = shap_values.values[0]
    input_row = input_df.iloc[0]

    # Build interpretation
    interpretation = interpret_prediction(probability, shap_row, input_df.iloc[0])

    if isinstance(interpretation, str):
        interpretation = interpretation.split("\n")

    # 5. Output message
    result = "✅ Customer is likely to churn." if prediction == 1 else "❌ Customer is not likely to churn."
    
        # SHAP explanation
    shap_values = explainer(input_df)
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    shap_path = os.path.join("static", "shap_output.png")
    plt.savefig(shap_path)
    plt.close()

    return render_template("index.html", result=result, probability=round(probability * 100, 2),shap_image="shap_output.png",interpretation=interpretation)

if __name__ == "__main__":
    app.run()
