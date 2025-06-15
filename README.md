# Groundwater Level Prediction Using Machine Learning (Telangana)

This project predicts groundwater levels for borewells in Telangana using a trained machine learning model. It uses spatial and hydrogeological features and exposes a simple web interface for predictions.

## 📂 Project Structure

GWL/
├── app.py                          # Flask web application
├── groundwater_rf_model.pkl        # Trained Random Forest model
├── GWL-MLCode.txt                  # Python code used for training
├── predicted_results.csv           # Sample predictions output
├── templates/
│   └── index.html                  # HTML template for UI
└── uploads/
    └── Telangana_Realistic_Borewell_Dataset.csv  # Groundwater dataset

## 🚀 How to Run the Project

### Requirements

Python 3.7+
Libraries: Flask, pandas, scikit-learn, joblib

### Installation

cd GWL
pip install -r requirements.txt  # If requirements.txt is not present:
pip install flask pandas scikit-learn joblib

### Start the Web App

python app.py

Then open your browser and go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 📊 Dataset

The dataset used: `Telangana_Realistic_Borewell_Dataset.csv` contains the following columns:

* Location - Region/district name
* Soil_Type - Categorical soil classification
* Rock_Type - Geological formation
* Aquifer_Type - Confined/unconfined
* Rainfall (mm) - Rainfall at the location
* Depth_to_Water_Level (m) - Groundwater level
* Seasonal_Fluctuation (m) - Water table variation
* Required_Bore_Depth (m) - Bore depth requirement (target)

## 🤖 Model

The machine learning model (groundwater_rf_model.pkl) is a Random Forest Regressor trained using hydrogeological and environmental features to predict the required borewell depth.

Input features used:

* Soil type
* Rock type
* Aquifer type
* Rainfall
* Seasonal fluctuation

Output:

Required borewell depth (in meters)

## 🖥️ UI Functionality

Upload or select borewell features from the web form
Model returns predicted borewell depth

## 📈 Sample Predictions

predicted_results.csv contains example outputs from the model, which can be used to validate performance or build a visualization.

## 🧠 Code Reference

GWL-MLCode.txt contains the Python code used to train the model (likely includes feature engineering, preprocessing, training, and model saving).

 📌 Credits

Dataset source: India-WRIS, CGWB, Telangana Groundwater Department
ML model: Scikit-learn RandomForestRegressor

This project demonstrates a practical application of ML and IoT-inspired datasets to aid sustainable water resource planning.
