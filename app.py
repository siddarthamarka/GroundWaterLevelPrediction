from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_FILE = "groundwater_rf_model.pkl"
PREDICTIONS_FILE = "predicted_results.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Column definitions
CATEGORICAL_COLUMNS = ["Location", "Soil_Type", "Rock_Type", "Aquifer_Type"]
NUMERICAL_COLUMNS = ["Rainfall (mm)", "Depth_to_Water_Level (m)", "Seasonal_Fluctuation (m)"]
TARGET_COLUMN = "Required_Bore_Depth (m)"

# Train model from uploaded CSV
def train_model(csv_file):
    df = pd.read_csv(csv_file)

    required_columns = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + [TARGET_COLUMN]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing required columns in the dataset.")

    X = df[CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS]
    y = df[TARGET_COLUMN]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERICAL_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(pipeline, f)

# Predict borewell depth from single input
def predict_depth(input_data):
    if not os.path.exists(MODEL_FILE):
        return None, "❌ Model not trained yet. Please upload a dataset first."

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict(input_data)[0]
    return round(prediction, 2), None

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    prediction = None

    if request.method == "POST":
        # Training with uploaded CSV
        if "file" in request.files:
            file = request.files["file"]
            if file.filename.endswith(".csv"):
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                try:
                    train_model(file_path)
                    message = "✅ Model trained successfully from uploaded dataset!"
                except Exception as e:
                    message = f"❌ Error in training model: {str(e)}"
            else:
                message = "❌ Invalid file format. Please upload a valid CSV file."

        # Predicting single input
        elif "Location" in request.form:
            try:
                form_data = {
                    "Location": request.form["Location"],
                    "Soil_Type": request.form["Soil_Type"],
                    "Rock_Type": request.form["Rock_Type"],
                    "Aquifer_Type": request.form["Aquifer_Type"],
                    "Rainfall (mm)": float(request.form["Rainfall"]),
                    "Depth_to_Water_Level (m)": float(request.form["Depth"]),
                    "Seasonal_Fluctuation (m)": float(request.form["Fluctuation"]),
                }

                input_df = pd.DataFrame([form_data])
                prediction, error = predict_depth(input_df)

                if error:
                    message = error
                else:
                    input_df["Predicted_Borewell_Depth (m)"] = prediction
                    input_df.to_csv(PREDICTIONS_FILE, index=False)
                    message = "✅ Prediction completed successfully!"

            except ValueError:
                message = "❌ Please enter valid numeric values for Rainfall, Depth, and Fluctuation."

    return render_template("index.html", message=message, prediction=prediction)

@app.route("/download")
def download():
    if os.path.exists(PREDICTIONS_FILE):
        return send_file(PREDICTIONS_FILE, as_attachment=True)
    else:
        return "❌ No prediction file found. Please generate a prediction first."

if __name__ == "__main__":
    app.run(debug=True)