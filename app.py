from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ==============================
# LOAD MODEL
# ==============================
data = joblib.load("C:\\Users\\vaishnavi potode\\OneDrive\\Desktop\\Projects\\laptop_price_prediction\\best_model.pkl")

model = data["model"]
scaler = data["scaler"]
encoders = data["encoders"]


# ==============================
# HOME PAGE
# ==============================
@app.route('/')
def home():
    return render_template("index.html")


# ==============================
# PREDICT ROUTE
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    
    # Get form data
    input_data = {
        'Company': request.form['Company'],
        'TypeName': request.form['TypeName'],
        'Ram': float(request.form['Ram']),
        'Weight': float(request.form['Weight']),
        'TouchScreen': int(request.form['TouchScreen']),
        'IPS': int(request.form['IPS']),
        'PPI': float(request.form['PPI']),
        'CPU_name': request.form['CPU_name'],
        'HDD': int(request.form['HDD']),
        'SSD': int(request.form['SSD']),
        'Gpu_brand': request.form['Gpu_brand'],
        'OpSys': request.form['OpSys']
    }

    df = pd.DataFrame([input_data])

    # Encode
    for col in ['Company', 'TypeName', 'OpSys', 'CPU_name', 'Gpu_brand']:
        le = encoders[col]
        
        if df[col][0] in le.classes_:
            df[col] = le.transform(df[col])
        else:
            df[col] = 0

    # Align columns
    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Scale
    scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(scaled)

    result = round(prediction[0], 2)

    return render_template("index.html", prediction_text=f"Estimated Price: ₹ {result}",form_data=input_data)


# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":
    app.run(debug=True)