import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import seaborn as sns

print(sns.__version__)

import sys
print(sys.executable)

warnings.filterwarnings("ignore")
plt.style.use("ggplot")
print("="*60)
print("HOUSE PRICE PREDICTION USING LINEAR REGRESSION")
print("="*60)

os.makedirs("Outputs", exist_ok=True)
os.makedirs("Screenshots", exist_ok=True)
print("Folders Created Successfully")

df = pd.read_csv("C:/Users/MANJIRI/OneDrive/Desktop/OIBSIP Internship/DataAnalytics_L2_HousePricePrediction/Housing.csv")
print("Dataset Loaded Successfully")

print(df.head())

print(df.shape)

print(df.columns.tolist())

df.info()

print(df.isnull().sum())

print("Duplicate Rows :", df.duplicated().sum())

df.drop_duplicates(inplace=True)
print(df.shape)

print(df.describe())

binary_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]
for col in binary_columns:
    df[col] = df[col].map({"yes": 1, "no": 0})
    
df["furnishingstatus"] = df["furnishingstatus"].map({
    "furnished": 2,
    "semi-furnished": 1,
    "unfurnished": 0
})

print(df.dtypes)

df.to_csv("Cleaned_Housing.csv", index=False)
print("Cleaned Dataset Saved Successfully")

print("="*60)
print("PROJECT SUMMARY")
print("="*60)
print("Total Houses :", len(df))
print("Average Price :", round(df["price"].mean(),2))
print("Maximum Price :", df["price"].max())
print("Minimum Price :", df["price"].min())
print("Average Area :", round(df["area"].mean(),2))


# ============================================================
# House Price Distribution
# ============================================================
plt.figure(figsize=(10,6))
sns.histplot(
    df["price"],
    bins=30,
    kde=True,
    color="steelblue"
)

plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/house_price_distribution.png", dpi=300)
plt.show()

# ============================================================
# Area Distribution
# ============================================================
plt.figure(figsize=(10,6))
sns.histplot(
    df["area"],
    bins=30,
    kde=True,
    color="green"
)

plt.title("Area Distribution")
plt.xlabel("Area")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/area_distribution.png", dpi=300)
plt.show()

# ============================================================
# Correlation Heatmap
# ============================================================
plt.figure(figsize=(12,8))
corr = df.corr(numeric_only=True)
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=300)
plt.show()

# ============================================================
# Price vs Area
# ============================================================
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="area",
    y="price",
    color="red"
)

plt.title("Price vs Area")
plt.xlabel("Area")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("outputs/price_vs_area.png", dpi=300)
plt.show()

# ============================================================
# Price vs Bedrooms
# ============================================================
plt.figure(figsize=(10,6))
sns.boxplot(
    data=df,
    x="bedrooms",
    y="price"
)

plt.title("Price vs Bedrooms")
plt.tight_layout()
plt.savefig("outputs/price_vs_bedrooms.png", dpi=300)
plt.show()

# ============================================================
# Price vs Bathrooms
# ============================================================
plt.figure(figsize=(10,6))
sns.boxplot(
    data=df,
    x="bathrooms",
    y="price"
)

plt.title("Price vs Bathrooms")
plt.tight_layout()
plt.savefig("outputs/price_vs_bathrooms.png", dpi=300)
plt.show()

# ============================================================
# Price vs Stories
# ============================================================
plt.figure(figsize=(10,6))
sns.boxplot(
    data=df,
    x="stories",
    y="price"
)

plt.title("Price vs Stories")
plt.tight_layout()
plt.savefig("outputs/price_vs_stories.png", dpi=300)
plt.show()

# ============================================================
# Furnishing Status
# ============================================================
plt.figure(figsize=(8,6))
sns.barplot(
    data=df,
    x="furnishingstatus",
    y="price",
    estimator=np.mean,
    errorbar=None,
    palette="viridis"
)

plt.title("Average Price by Furnishing Status")
plt.xlabel("Furnishing Status")
plt.ylabel("Average Price")
plt.tight_layout()
plt.savefig("outputs/furnishing_status.png", dpi=300)
plt.show()

# ============================================================
# Pair Plot
# ============================================================
sns.pairplot(
    df[
        [
            "price",
            "area",
            "bedrooms",
            "bathrooms",
            "stories"
        ]
    ]
)

plt.savefig("outputs/pairplot.png", dpi=300)
plt.show()

# ============================================================
# EDA Summary
# ============================================================
print("="*60)
print("EDA SUMMARY")
print("="*60)
print("Average Price :", round(df["price"].mean(),2))
print("Maximum Price :", df["price"].max())
print("Minimum Price :", df["price"].min())
print("Average Area :", round(df["area"].mean(),2))
print("Average Bedrooms :", round(df["bedrooms"].mean(),2))
print("Average Bathrooms :", round(df["bathrooms"].mean(),2))
print("Average Stories :", round(df["stories"].mean(),2))

# ============================================================
# Feature Selection
# ============================================================
X = df.drop("price", axis=1)
y = df["price"]
print("Features Shape :", X.shape)
print("Target Shape :", y.shape)

# ============================================================
# Feature Names
# ============================================================
print("\nFeatures Used for Prediction:\n")
print(X.columns.tolist())

# ============================================================
# Train Test Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("Training Data :", X_train.shape)
print("Testing Data :", X_test.shape)

# ============================================================
# Linear Regression Model
# ============================================================
model = LinearRegression()
print("Model Created Successfully")

# ============================================================
# Train Model
# ============================================================
model.fit(X_train, y_train)
print("Model Trained Successfully")

# ============================================================
# Model Coefficients
# ============================================================
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print(coefficients)

# ============================================================
# Intercept
# ============================================================
print("Intercept :", model.intercept_)

# ============================================================
# Predictions
# ============================================================
y_pred = model.predict(X_test)
print("Predictions Generated Successfully")

# ============================================================
# STEP 35: Comparison Actual vs Predicted
# ============================================================
comparison = pd.DataFrame({

    "Actual Price": y_test.values,

    "Predicted Price": y_pred

})
print(comparison.head(10))

# ============================================================
# Save Prediction Results
# ============================================================
comparison.to_csv(
    "Predicted_House_Prices.csv",
    index=False
)
print("Prediction File Saved Successfully")

# ============================================================
# Feature Importance
# ============================================================
importance = coefficients.sort_values(
    by="Coefficient",
    ascending=False
)
print(importance)

# ============================================================
# Feature Importance Plot
# ============================================================
plt.figure(figsize=(10,6))
sns.barplot(
    data=importance,
    x="Coefficient",
    y="Feature",
    palette="viridis"
)

plt.title("Feature Importance")
plt.tight_layout()
plt.savefig(
    "outputs/feature_importance.png",
    dpi=300
)
plt.show()

# ============================================================
# Training Summary
# ============================================================
print("="*60)
print("MODEL TRAINING COMPLETED")
print("="*60)
print("Training Samples :", len(X_train))
print("Testing Samples :", len(X_test))
print("Number of Features :", X.shape[1])

# ============================================================
# Import Evaluation Metrics
# ============================================================
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
print("Evaluation Metrics Imported Successfully")

# ============================================================
# Mean Absolute Error
# ============================================================
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", round(mae,2))

# ============================================================
# Mean Squared Error
# ============================================================
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", round(mse,2))

# ============================================================
# Root Mean Squared Error
# ============================================================
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", round(rmse,2))

# ============================================================
# R² Score
# ============================================================
r2 = r2_score(y_test, y_pred)
print("R² Score:", round(r2,4))

# ============================================================
# Actual vs Predicted
# ============================================================
plt.figure(figsize=(8,6))
plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig(
    "outputs/actual_vs_predicted.png",
    dpi=300
)
plt.show()

# ============================================================
# Residual Plot
# ============================================================
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
plt.scatter(
    y_pred,
    residuals,
    alpha=0.7
)
plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(
    "outputs/residual_plot.png",
    dpi=300
)
plt.show()


# ============================================================
# Model Performance Summary
# ============================================================
print("="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# ============================================================
# Business Insights
# ============================================================
print("\nBUSINESS INSIGHTS")
print("="*60)
print("""
1. House area has a strong influence on selling price.
2. Additional bathrooms and bedrooms generally increase property value.
3. Features such as air conditioning, parking, and main road access
   contribute positively to house prices.
4. The trained Linear Regression model can provide quick price estimates
   for similar properties.
5. This approach can support pricing decisions in the real estate market.
""")

# ============================================================
# Save Evaluation Results
# ============================================================
evaluation = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2 Score"],
    "Value": [mae, mse, rmse, r2]
})
evaluation.to_csv(
    "Model_Evaluation.csv",
    index=False
)
print("Model Evaluation Saved Successfully")


# ============================================================
# Project Summary
# ============================================================
print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)
print("Total Houses :", len(df))
print("Training Samples :", len(X_train))
print("Testing Samples :", len(X_test))
print("Number of Features :", X.shape[1])
print("Model : Linear Regression")
print("Best Metric : R² =", round(r2,4))
print("\nThank you for using this House Price Prediction Project!")






















































