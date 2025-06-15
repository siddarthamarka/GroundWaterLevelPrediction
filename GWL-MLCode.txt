import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("/content/Telangana_Realistic_Borewell_Dataset.csv")

# Features and target
X = df.drop(columns=["Required_Bore_Depth (m)"])
y = df["Required_Bore_Depth (m)"]

# Categorical and numerical columns
categorical_features = ["Location", "Soil_Type", "Rock_Type", "Aquifer_Type"]
numerical_features = ["Rainfall (mm)", "Depth_to_Water_Level (m)", "Seasonal_Fluctuation (m)"]

# Preprocessor: scale numerics, one-hot encode categoricals
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Split data (without stratifying)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name} Results:")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R² Score: {r2:.3f}")

    # Plot actual vs predicted values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect Prediction")
    plt.title(f"Actual vs Predicted Bore Depth ({name})")
    plt.xlabel("Actual Bore Depth (m)")
    plt.ylabel("Predicted Bore Depth (m)")
    plt.legend()
    plt.show()

# Average Required_Bore_Depth (m) for each district (Location)
avg_depth_by_location = df.groupby("Location")["Required_Bore_Depth (m)"].mean().sort_values(ascending=False)

# Plot average bore depth by location (district)
plt.figure(figsize=(12, 6))
avg_depth_by_location.plot(kind='bar', color='green')
plt.title('Average Required Bore Depth by District Location')
plt.xlabel('District Location')
plt.ylabel('Average Required Bore Depth (m)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Correlation matrix for numerical features
correlation_matrix = df[numerical_features + ["Required_Bore_Depth (m)"]].corr()

# Print correlation values
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="Reds", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# Group by Soil_Type and Rock_Type, then calculate average Required_Bore_Depth
avg_depth_by_soil_rock = df.groupby(["Soil_Type", "Rock_Type"])["Required_Bore_Depth (m)"].mean().sort_values(ascending=False)

# Display the result
print("Average Bore Depth by Soil Type and Rock Type:")
print(avg_depth_by_soil_rock)

# Average Required_Bore_Depth (m) for each district (Location)
avg_depth_by_location = df.groupby("Location")["Required_Bore_Depth (m)"].mean().sort_values(ascending=False)

# State-level average
avg_depth_state = df["Required_Bore_Depth (m)"].mean()
print(f"\nAverage Required Bore Depth for Telangana (State): {avg_depth_state:.2f} meters")

# Print district-wise average required bore depth
print("District-wise Average Required Bore Depth:")
for district, avg_depth in avg_depth_by_location.items():
    print(f"{district}: {avg_depth:.2f} meters")

# Group by Location (district) and Soil_Type, then calculate average Required_Bore_Depth
avg_depth_by_location_soil = df.groupby(["Location", "Soil_Type"])["Required_Bore_Depth (m)"].mean().sort_values(ascending=False)

# Print district-wise average required bore depth considering Soil_Type
print("District-wise Average Required Bore Depth considering Soil Type:")
for (district, soil_type), avg_depth in avg_depth_by_location_soil.items():
    print(f"{district} ({soil_type}): {avg_depth:.2f} meters")