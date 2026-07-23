# Import Libraries

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
plt.style.use("ggplot")
print("="*60)
print("Customer Segmentation Analysis")
print("="*60)

# ==============================================================
# Load Dataset
# ==============================================================
df = pd.read_csv("C:/Users/MANJIRI/OneDrive/Desktop/OIBSIP Internship/DataAnalytics_L1_CustomerSegmentation/Online Retail.csv", encoding="ISO-8859-1")
print("\nDataset Loaded Successfully!")

# ==============================================================
# Create folders
# ==============================================================
os.makedirs("outputs", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)
print("Project folders created successfully.")

# ==============================================================
# First Five Rows
# ==============================================================
print("\nFirst Five Records\n")
print(df.head())

# ==============================================================
# Dataset Shape
# ==============================================================
print("\nShape of Dataset")
print(df.shape)

# ==============================================================
# Columns
# ==============================================================
print("\nColumns\n")
print(df.columns.tolist())

# ==============================================================
# Dataset Information
# ==============================================================
print("\nDataset Information\n")
df.info()

# ==============================================================
# Missing Values
# ==============================================================
print("\nMissing Values\n")
print(df.isnull().sum())

# ==============================================================
# Duplicate Records
# ==============================================================
duplicates = df.duplicated().sum()
print("\nDuplicate Rows :", duplicates)

# ==============================================================
# Remove Duplicates
# ==============================================================
df.drop_duplicates(inplace=True)
print("\nShape After Removing Duplicates")
print(df.shape)

# ==============================================================
# Remove Missing CustomerID
# ==============================================================
df = df.dropna(subset=["CustomerID"])
print("\nShape After Removing Missing Customer IDs")
print(df.shape)

# ==============================================================
# Convert Data Types
# ==============================================================
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    format="%d-%m-%Y %H:%M")

# ==============================================================
# Remove Invalid Quantity & Price
# ==============================================================
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
print("\nShape After Cleaning")
print(df.shape)

# ==============================================================
# Total Amount
# ==============================================================
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
print("\nTotalAmount Column Created")

# ==============================================================
# Statistics
# ==============================================================
print("\nDescriptive Statistics\n")
print(df.describe())

# ==============================================================
# Data Types
# ==============================================================
print("\nData Types\n")
print(df.dtypes)

# ==============================================================
# Save Clean Dataset
# ==============================================================
df.to_csv(
    "Cleaned_Online_Retail.csv",
    index=False
)
print("\nCleaned Dataset Saved Successfully!")

# ==============================================================
# Summary
# ==============================================================
print("\nPROJECT SUMMARY")
print("="*50)

print("Total Records :", len(df))
print("Total Customers :", df["CustomerID"].nunique())
print("Total Countries :", df["Country"].nunique())
print("Total Products :", df["StockCode"].nunique())
print("Total Sales : £", round(df["TotalAmount"].sum(),2))

# ==============================================================
# Top 10 Countries by Sales
# ==============================================================
country_sales = (
    df.groupby("Country")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,6))
sns.barplot(
    x=country_sales.values,
    y=country_sales.index,
    palette="viridis"
)

plt.title("Top 10 Countries by Sales", fontsize=15)
plt.xlabel("Total Sales (£)")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig("outputs/top_countries_sales.png", dpi=300)
plt.show()

# ==============================================================
# Top Customers
# ==============================================================
top_customers = (
    df.groupby("CustomerID")["TotalAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,6))
sns.barplot(
    x=top_customers.index.astype(str),
    y=top_customers.values,
    palette="Set2"
)

plt.xticks(rotation=45)
plt.title("Top 10 Customers by Sales")
plt.xlabel("Customer ID")
plt.ylabel("Sales (£)")
plt.tight_layout()
plt.savefig("outputs/top_customers.png", dpi=300)
plt.show()

# ==============================================================
# Top Selling Products
# ==============================================================
top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,6))
sns.barplot(
    x=top_products.values,
    y=top_products.index,
    palette="rocket"
)

plt.title("Top 10 Selling Products")
plt.xlabel("Quantity Sold")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig("outputs/top_products.png", dpi=300)
plt.show()

# ==============================================================
# Sales Distribution
# ==============================================================
plt.figure(figsize=(10,6))
sns.histplot(
    df["TotalAmount"],
    bins=50,
    kde=True
)

plt.title("Sales Distribution")
plt.xlabel("Sales (£)")
plt.tight_layout()
plt.savefig("outputs/sales_distribution.png", dpi=300)
plt.show()

# ==============================================================
# Quantity Distribution
# ==============================================================
plt.figure(figsize=(10,6))
sns.histplot(
    df["Quantity"],
    bins=40,
    color="green"
)

plt.title("Quantity Distribution")
plt.tight_layout()
plt.savefig("outputs/quantity_distribution.png", dpi=300)
plt.show()

# ==============================================================
# Unit Price Distribution
# ==============================================================
plt.figure(figsize=(10,6))
sns.histplot(
    df["UnitPrice"],
    bins=40,
    color="orange"
)

plt.title("Unit Price Distribution")
plt.tight_layout()
plt.savefig("outputs/unitprice_distribution.png", dpi=300)
plt.show()

# ==============================================================
# Monthly Sales
# ==============================================================
monthly_sales = (
    df
    .set_index("InvoiceDate")
    .resample("M")["TotalAmount"]
    .sum()
)

plt.figure(figsize=(14,6))
monthly_sales.plot(
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.ylabel("Sales (£)")
plt.tight_layout()
plt.savefig("outputs/monthly_sales.png", dpi=300)
plt.show()

# ==============================================================
# Correlation Heatmap
# ==============================================================
plt.figure(figsize=(8,6))
corr = df[
    ["Quantity","UnitPrice","TotalAmount"]
].corr()
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/heatmap.png", dpi=300)
plt.show()

# ==============================================================
# Boxplots
# ==============================================================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.boxplot(y=df["Quantity"])
plt.title("Quantity")
plt.subplot(1,2,2)
sns.boxplot(y=df["TotalAmount"])
plt.title("Sales")
plt.tight_layout()
plt.savefig("outputs/boxplots.png", dpi=300)
plt.show()

# ==============================================================
# EDA Summary
# ==============================================================
print("\nEDA SUMMARY")
print("="*50)
print("Average Sales : £", round(df["TotalAmount"].mean(),2))
print("Maximum Sale : £", round(df["TotalAmount"].max(),2))
print("Minimum Sale : £", round(df["TotalAmount"].min(),2))
print("Average Quantity :", round(df["Quantity"].mean(),2))
print("Average Unit Price : £", round(df["UnitPrice"].mean(),2))


# ==============================================================
# Create Reference Date
# ==============================================================
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
print("Reference Date :", reference_date)

# ==============================================================
# Create RFM Table
# ==============================================================
rfm = df.groupby("CustomerID").agg({

    "InvoiceDate": lambda x: (reference_date - x.max()).days,

    "ï»¿InvoiceNo": "nunique",

    "TotalAmount": "sum"

})
rfm.columns = ["Recency", "Frequency", "Monetary"]
print(rfm.head())

# ==============================================================
# RFM Summary
# ==============================================================
print("\nRFM Summary")
print(rfm.describe())

# ==============================================================
# Missing Values
# ==============================================================
print(rfm.isnull().sum())

# ==============================================================
# Histograms
# ==============================================================
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
sns.histplot(rfm["Recency"], bins=30, color="blue")
plt.title("Recency")

plt.subplot(1,3,2)
sns.histplot(rfm["Frequency"], bins=30, color="green")
plt.title("Frequency")

plt.subplot(1,3,3)
sns.histplot(rfm["Monetary"], bins=30, color="red")
plt.title("Monetary")

plt.tight_layout()
plt.savefig("outputs/rfm_histograms.png", dpi=300)
plt.show()

# ==============================================================
# STEP 34: Boxplots
# ==============================================================
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
sns.boxplot(y=rfm["Recency"])

plt.subplot(1,3,2)
sns.boxplot(y=rfm["Frequency"])

plt.subplot(1,3,3)
sns.boxplot(y=rfm["Monetary"])

plt.tight_layout()
plt.savefig("outputs/rfm_boxplots.png", dpi=300)
plt.show()

# ==============================================================
# StandardScaler
# ==============================================================
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)
print("Data Standardized Successfully")

# ==============================================================
# Elbow Method
# ==============================================================
wcss = []
for i in range(1,11):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(rfm_scaled)

    wcss.append(model.inertia_)

# ==============================================================
# Plot Elbow Curve
# ==============================================================
plt.figure(figsize=(8,5))
plt.plot(
    range(1,11),
    wcss,
    marker="o",
    linewidth=2
)

plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.savefig("outputs/elbow_method.png", dpi=300)
plt.show()

# ==============================================================
# KMeans
# ==============================================================
kmeans = KMeans(

    n_clusters=4,

    random_state=42,

    n_init=10

)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
print("KMeans Model Created Successfully")

# ==============================================================
# Cluster Count
# ==============================================================
print(rfm["Cluster"].value_counts())

# ==============================================================
# Save Dataset
# ==============================================================
rfm.to_csv(

    "Clustered_Customers.csv"

)
print("Clustered Dataset Saved Successfully")

# ==============================================================
# Cluster Summary
# ==============================================================
cluster_summary = rfm.groupby("Cluster").mean()
print(cluster_summary)

# ==============================================================
# Customers Per Cluster
# ==============================================================
plt.figure(figsize=(8,5))
sns.countplot(

    x="Cluster",

    data=rfm,

    palette="Set2"

)
plt.title("Customers in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(

    "outputs/customers_per_cluster.png",

    dpi=300

)
plt.show()

# ==============================================================
# Frequency vs Monetary
# ==============================================================
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=rfm,
    x="Frequency",
    y="Monetary",
    hue="Cluster",
    palette="Set1",
    s=80
)

plt.title("Customer Segments (Frequency vs Monetary)")
plt.xlabel("Frequency")
plt.ylabel("Monetary (£)")
plt.tight_layout()
plt.savefig("outputs/frequency_vs_monetary.png", dpi=300)
plt.show()

# ==============================================================
# Recency vs Monetary
# ==============================================================
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=rfm,
    x="Recency",
    y="Monetary",
    hue="Cluster",
    palette="Dark2",
    s=80
)

plt.title("Customer Segments (Recency vs Monetary)")
plt.xlabel("Recency")
plt.ylabel("Monetary (£)")
plt.tight_layout()
plt.savefig("outputs/recency_vs_monetary.png", dpi=300)
plt.show()

# ==============================================================
# Pair Plot
# ==============================================================
sns.pairplot(
    rfm,
    vars=["Recency","Frequency","Monetary"],
    hue="Cluster",
    palette="Set2"
)
plt.savefig("outputs/pairplot.png", dpi=300)
plt.show()

# ==============================================================
# Cluster Means
# ==============================================================
cluster_profile = rfm.groupby("Cluster").agg({

    "Recency":"mean",
    "Frequency":"mean",
    "Monetary":"mean"

}).round(2)
print(cluster_profile)

# ==============================================================
# Save Cluster Profile
# ==============================================================
cluster_profile.to_csv("Cluster_Profile.csv")
print("Cluster Profile Saved Successfully")

# ==============================================================
# Cluster Heatmap
# ==============================================================
plt.figure(figsize=(8,5))
sns.heatmap(

    cluster_profile,

    annot=True,

    cmap="YlGnBu"

)
plt.title("Cluster Profile Heatmap")
plt.tight_layout()
plt.savefig("outputs/cluster_heatmap.png", dpi=300)
plt.show()

# ==============================================================
# Cluster Comparison
# ==============================================================
cluster_profile.plot(

    kind="bar",

    figsize=(10,6)
)
plt.title("Cluster Comparison")
plt.ylabel("Average Value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/cluster_comparison.png", dpi=300)
plt.show()

# ==============================================================
# Business Interpretation
# ==============================================================
print("\nBUSINESS INTERPRETATION")
print("="*60)
for cluster in sorted(rfm["Cluster"].unique()):

    print(f"\nCluster {cluster}")

    print(cluster_profile.loc[cluster])

    print("-"*50)
    
# ==============================================================
# Marketing Recommendations
# ==============================================================
print("\nMARKETING RECOMMENDATIONS")
print("="*60)
print("""

Cluster 0
---------
High-value customers.
Offer loyalty rewards, premium memberships, and exclusive discounts.

Cluster 1
---------
Regular customers.
Increase purchase frequency through personalized offers.

Cluster 2
---------
Inactive customers.
Run email campaigns, coupons, and re-engagement promotions.

Cluster 3
---------
New or low-spending customers.
Provide welcome offers and cross-selling opportunities.
""")

# ==============================================================
# Conclusion
# ==============================================================
print("\nPROJECT COMPLETED SUCCESSFULLY")
print("="*60)
print("Total Customers :", rfm.shape[0])
print("Number of Clusters :", rfm["Cluster"].nunique())
print("Average Monetary Value : £", round(rfm["Monetary"].mean(),2))
print("Highest Spending Customer : £", round(rfm["Monetary"].max(),2))
print("Lowest Spending Customer : £", round(rfm["Monetary"].min(),2))

### Business Recommendations
#- Loyalty Programs
#- Personalized Marketing
#- Discount Campaigns
#- Cross Selling
#- Customer Retention Strategies
