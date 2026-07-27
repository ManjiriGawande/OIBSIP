Google Play Store Analysis

Oasis Infobyte Data Analytics Internship – Level 2 Task 4**

### Project Overview

The Google Play Store Analysis project focuses on analyzing application data from the Google Play Store to uncover meaningful insights into app ratings, installs, reviews, pricing, and category performance.
Using Python and data analysis libraries, the project performs **data cleaning, preprocessing, exploratory data analysis (EDA), visualization, and business insight generation** to help understand factors influencing app popularity and user engagement.

### Objectives
- Analyze Google Play Store application data.
- Clean and preprocess real-world datasets.
- Perform Exploratory Data Analysis (EDA).
- Identify trends in app categories, ratings, installs, reviews, and pricing.
- Generate actionable business insights using data visualization.

### Dataset
Dataset Name: Google Play Store Apps
Source: https://www.kaggle.com/datasets/lava18/google-play-store-apps
Dataset File : googleplaystore.csv

### Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Spyder IDE

### Project Structure

DataAnalytics-L2-GooglePlayStoreAnalysis/
│
├── google_play_store_analysis.py
├── googleplaystore.csv
├── Cleaned_GooglePlayStore.csv
├── Google_PlayStore_Final.csv
├── Category_Analysis.csv
├── Top_Rated_Apps.csv
├── Top_Installed_Apps.csv
├── Top_Reviewed_Apps.csv
├── README.md
├── requirements.txt
│
├── outputs/
│   ├── rating_distribution.png
│   ├── top_categories.png
│   ├── top_installs.png
│   ├── free_vs_paid.png
│   ├── category_rating.png
│   ├── reviews_distribution.png
│   ├── price_distribution.png
│   └── correlation_heatmap.png
│
└── screenshots/

### Project Workflow

## 1️⃣ Data Collection
- Imported the Google Play Store dataset.
- Loaded data using Pandas.

## 2️⃣ Data Cleaning
- Removed duplicate records.
- Handled missing values.
- Converted Reviews, Rating, Installs, and Price into numeric format.
- Processed the Size column for analysis.

## 3️⃣ Exploratory Data Analysis (EDA)
- Rating Distribution
- Top Categories
- Top Installed Categories
- Free vs Paid Apps
- Category-wise Average Rating
- Reviews Distribution
- Price Distribution
- Correlation Heatmap

## 4️⃣ Data Export
Generated:
- Cleaned Dataset
- Category Analysis Report
- Top Rated Apps
- Top Installed Apps
- Top Reviewed Apps

## 5️⃣ Business Insights
Generated insights to understand user preferences, category performance, and application popularity.

### Visualizations
The project includes the following charts:
- Rating Distribution
- Top App Categories
- Category-wise Installs
- Average Rating by Category
- Free vs Paid Apps
- Reviews Distribution
- Price Distribution
- Correlation Heatmap

### Output Files
After execution, the project generates:
- Cleaned_GooglePlayStore.csv
- Google_PlayStore_Final.csv
- Category_Analysis.csv
- Top_Rated_Apps.csv
- Top_Installed_Apps.csv
- Top_Reviewed_Apps.csv

### Key Business Insights
- Free applications dominate the Google Play Store ecosystem.
- Family, Tools, and Game categories contain the highest number of applications.
- Highly rated applications generally receive more installs and better user engagement.
- Reviews are strongly associated with application popularity.
- Paid applications account for only a small percentage of the overall Play Store.
- Developers should focus on improving user experience and ratings to increase app visibility and downloads.

### Sample Outputs
The project generates visualizations including:
- Rating Distribution
- Top Categories
- Free vs Paid Apps
- Category-wise Average Rating
- Correlation Heatmap
- Reviews Distribution
- Price Distribution

### Future Enhancements
- Interactive Dashboard using Power BI or Tableau
- Sentiment Analysis on User Reviews
- Machine Learning-based App Success Prediction
- Recommendation System for App Categories
- Interactive Dashboard using Streamlit

### Requirements
Install the required Python libraries:
pip install -r requirements.txt

### How to Run
python google_play_store_analysis.py
or
run the script directly using **Spyder IDE**.

### Author
Manjiri Gawande
Data Analytics Intern  
Oasis Infobyte

