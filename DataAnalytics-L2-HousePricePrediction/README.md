House Price Prediction using Linear Regression

Oasis Infobyte Data Analytics Internship

### Project Overview
This project predicts house prices using **Linear Regression**, one of the most widely used supervised machine learning algorithms. The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, model building, performance evaluation, and visualization.

### Objective
The objective of this project is to build a machine learning model that can estimate house prices based on property features such as:
- Area
- Bedrooms
- Bathrooms
- Stories
- Parking
- Main Road Access
- Air Conditioning
- Furnishing Status
- Basement
- Guest Room
- Preferred Area
- Hot Water Heating

### Dataset
Dataset Name: Housing Price Dataset
Dataset File: Housing.csv
The dataset contains information about residential properties and their corresponding selling prices.

### Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Spyder IDE

### Machine Learning Algorithm
- Linear Regression

### Project Workflow

1. Data Loading
- Import dataset
- Check dataset structure

2. Data Cleaning
- Remove duplicate records
- Encode categorical variables
- Verify data types

3. Exploratory Data Analysis
- House Price Distribution
- Area Distribution
- Correlation Heatmap
- Price vs Area
- Price vs Bedrooms
- Price vs Bathrooms
- Price vs Stories
- Furnishing Status Analysis

4. Feature Engineering
- Encode categorical features
- Select independent and dependent variables

5. Model Building
- Train-Test Split
- Linear Regression Model

6. Model Evaluation
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

7. Visualization
- Feature Importance
- Actual vs Predicted Prices
- Residual Plot

8. Business Insights
- Identify major factors affecting house prices
- Predict house prices for future properties

### Project Structure
DataAnalytics-L2-HousePricePrediction/
│
├── house_price_prediction.py
├── Housing.csv
├── Cleaned_Housing.csv
├── Predicted_House_Prices.csv
├── Model_Evaluation.csv
├── README.md
├── requirements.txt
├── outputs/
│   house_price_distribution.png
│   area_distribution.png
│   correlation_heatmap.png
│   price_vs_area.png
│   price_vs_bedrooms.png
│   price_vs_bathrooms.png
│   price_vs_stories.png
│   furnishing_status.png
│   pairplot.png
│   feature_importance.png
│   actual_vs_predicted.png
│   residual_plot.png
│
└── screenshots/

### Results
The Linear Regression model successfully predicts house prices using property features.
The model was evaluated using:
- MAE
- MSE
- RMSE
- R² Score
These evaluation metrics help measure prediction accuracy and overall model performance.

### Business Insights
- Property area has a significant impact on house price.
- Houses with more bathrooms and bedrooms generally have higher prices.
- Air conditioning, parking, and main road access positively influence property value.
- The model can assist buyers, sellers, and real estate agencies in estimating house prices.

### Sample Outputs
- House Price Distribution
- Correlation Heatmap
- Feature Importance
- Actual vs Predicted Prices
- Residual Plot

### Future Improvements
- Random Forest Regressor
- XGBoost Regressor
- Hyperparameter Tuning
- Cross Validation
- Model Deployment using Streamlit or Flask

### Author
Manjiri Gawande
Data Analytics Intern
Oasis Infobyte


