# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load your dataset
# Assuming you have a DataFrame called 'houses' with relevant columns
# Replace 'your_dataset.csv' with the actual file path or use your preferred method to load data
houses = pd.read_csv('dataset.csv')

# 1. Data Cleaning & Preprocessing
# Handling Missing Values
houses = houses.dropna()

# Handling Categorical Data
label_encoder = LabelEncoder()
houses['categorical_column'] = label_encoder.fit_transform(houses['categorical_column'])

# Normalization/Standardization
scaler = StandardScaler()
houses[['numerical_column1', 'numerical_column2']] = scaler.fit_transform(houses[['numerical_column1', 'numerical_column2']])

# Outlier Detection
# You can use a method like IQR (Interquartile Range) or Z-score to detect and handle outliers

# 2. Finding the Optimal Value of K
# Elbow Method
sse = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(houses)
    sse.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(2, 11), sse, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances (SSE)')
plt.title('Elbow Method for Optimal k')
plt.show()

# Silhouette Score
sil_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(houses)
    silhouette_avg = silhouette_score(houses, cluster_labels)
    sil_scores.append(silhouette_avg)

# Plot silhouette scores
plt.plot(range(2, 11), sil_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.show()

# Choose the optimal k based on the elbow or silhouette method
optimal_k = 3  # Replace with your chosen value

# 3. Clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
houses['cluster'] = kmeans.fit_predict(houses)

# 4. Storing Cluster Information
# Assuming you want to save the clustered data to a new CSV file
houses.to_csv('clustered_houses.csv', index=False)
