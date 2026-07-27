import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

plt.style.use("ggplot")

print("="*60)
print("GOOGLE PLAY STORE ANALYSIS")
print("="*60)

os.makedirs("outputs", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)
print("Folders Created Successfully")

df = pd.read_csv("C:/Users/MANJIRI/OneDrive/Desktop/OIBSIP Internship/DataAnalytics-L2-GooglePlayStoreAnalysis/googleplaystore.csv")
print("Dataset Loaded Successfully")

print(df.head())

print("Dataset Shape :", df.shape)

print(df.columns.tolist())

df.info()

print(df.isnull().sum())

print("Duplicate Rows :", df.duplicated().sum())

df.drop_duplicates(inplace=True)
print(df.shape)

df = df.dropna(subset=["Rating"])

df["Installs"] = (
    df["Installs"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("+", "", regex=False)
)
df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")

df["Price"] = (
    df["Price"]
    .astype(str)
    .str.replace("$", "", regex=False)
)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

df["Reviews"] = pd.to_numeric(
    df["Reviews"],
    errors="coerce"
)

df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
)

def convert_size(value):

    value = str(value)

    if value.endswith("M"):
        return float(value[:-1]) * 1024

    elif value.endswith("k"):
        return float(value[:-1])

    else:
        return np.nan
df["Size_KB"] = df["Size"].apply(convert_size)

print(df.dtypes)

print(df.describe())

df.to_csv(
    "Cleaned_GooglePlayStore.csv",
    index=False
)
print("Clean Dataset Saved Successfully")

print("="*60)
print("PROJECT SUMMARY")
print("="*60)
print("Total Apps :", len(df))
print("Average Rating :", round(df["Rating"].mean(),2))
print("Maximum Rating :", df["Rating"].max())
print("Minimum Rating :", df["Rating"].min())
print("Average Reviews :", round(df["Reviews"].mean(),2))

# ============================================================
# Rating Distribution
# ============================================================
plt.figure(figsize=(8,5))
sns.histplot(df["Rating"], bins=20, kde=True)

plt.title("Distribution of App Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Apps")
plt.tight_layout()
plt.savefig("outputs/rating_distribution.png", dpi=300)
plt.show()

# ============================================================
# Top Categories
# ============================================================
plt.figure(figsize=(10,6))
category_count = df["Category"].value_counts().head(10)

sns.barplot(
    x=category_count.values,
    y=category_count.index
)

plt.title("Top 10 App Categories")
plt.tight_layout()
plt.savefig("outputs/top_categories.png", dpi=300)
plt.show()

# ============================================================
# Top Categories by Installs
# ============================================================
install_category = df.groupby("Category")["Installs"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(
    x=install_category.values,
    y=install_category.index
)

plt.title("Top Categories by Total Installs")
plt.tight_layout()
plt.savefig("outputs/top_installs.png", dpi=300)
plt.show()

# ============================================================
# Free vs Paid Apps
# ============================================================
plt.figure(figsize=(6,6))
df["Type"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.ylabel("")
plt.title("Free vs Paid Apps")
plt.tight_layout()
plt.savefig("outputs/free_vs_paid.png", dpi=300)
plt.show()

# ============================================================
# Average Rating by Category
# ============================================================
avg_rating = (
    df.groupby("Category")["Rating"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
plt.figure(figsize=(10,6))
sns.barplot(
    x=avg_rating.values,
    y=avg_rating.index
)

plt.title("Top Categories by Average Rating")
plt.tight_layout()
plt.savefig("outputs/category_rating.png", dpi=300)
plt.show()

# ============================================================
# Reviews Distribution
# ============================================================
plt.figure(figsize=(8,5))
sns.histplot(df["Reviews"], bins=30)
plt.title("Reviews Distribution")
plt.tight_layout()
plt.savefig("outputs/reviews_distribution.png", dpi=300)
plt.show()

# ============================================================
# Price Distribution
# ============================================================
plt.figure(figsize=(8,5))
sns.histplot(df["Price"], bins=20)
plt.title("Price Distribution")
plt.tight_layout()
plt.savefig("outputs/price_distribution.png", dpi=300)
plt.show()

# ============================================================
# Correlation Heatmap
# ============================================================
numeric = df.select_dtypes(include=np.number)
plt.figure(figsize=(8,6))
sns.heatmap(
    numeric.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=300)
plt.show()

# ============================================================
# Top Rated Apps
# ============================================================
top_apps = (
    df[["App", "Rating"]]
    .sort_values(by="Rating", ascending=False)
    .drop_duplicates()
    .head(10)
)
print(top_apps)
top_apps.to_csv(
    "Top_Rated_Apps.csv",
    index=False
)

# ============================================================
# Category Analysis
# ============================================================
category_analysis = df.groupby("Category").agg({

    "Rating":"mean",

    "Reviews":"mean",

    "Installs":"sum"

}).round(2)
print(category_analysis)
category_analysis.to_csv(
    "Category_Analysis.csv"
)

# ============================================================
# EDA Summary
# ============================================================
print("="*60)
print("EDA COMPLETED SUCCESSFULLY")
print("="*60)
print("Total Categories :", df["Category"].nunique())
print("Highest Rated Category :",
      avg_rating.idxmax())
print("Average Rating :",
      round(df["Rating"].mean(),2))

# ============================================================
# Top 10 Most Installed Apps
# ============================================================
top_installed = (
    df[["App", "Installs"]]
    .sort_values(by="Installs", ascending=False)
    .drop_duplicates()
    .head(10)
)
print("\nTop 10 Most Installed Apps")
print(top_installed)
top_installed.to_csv(
    "Top_Installed_Apps.csv",
    index=False
)

# ============================================================
# Top Reviewed Apps
# ============================================================
top_reviews = (
    df[["App", "Reviews"]]
    .sort_values(by="Reviews", ascending=False)
    .drop_duplicates()
    .head(10)
)
print("\nTop Reviewed Apps")
print(top_reviews)
top_reviews.to_csv(
    "Top_Reviewed_Apps.csv",
    index=False
)

# ============================================================
# Business Insights
# ============================================================
print("\n" + "="*60)
print("BUSINESS INSIGHTS")
print("="*60)
print("""
1. Free applications dominate the Google Play Store.
2. Family, Tools and Game categories contain the highest number of apps.
3. Apps with higher ratings generally receive more installs and better user engagement.
4. A small number of applications contribute to the majority of total installs.
5. Paid applications represent only a small percentage of the Play Store.
6. User reviews provide a strong indicator of an application's popularity.
7. Developers should focus on app quality and user satisfaction to improve ratings.
""")

# ============================================================
# Save Final Dataset
# ============================================================
df.to_csv(
    "Google_PlayStore_Final.csv",
    index=False
)
print("Final Dataset Saved Successfully")

# ============================================================
# Project Summary
# ============================================================
print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)
print("Total Apps :", len(df))
print("Total Categories :", df["Category"].nunique())
print("Average Rating :", round(df["Rating"].mean(),2))
print("Highest Rating :", df["Rating"].max())
print("Lowest Rating :", df["Rating"].min())
print("Average Reviews :", int(df["Reviews"].mean()))
print("\nGoogle Play Store Analysis Completed Successfully!")



