import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Select relevant features and standardize them."""
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler

def find_optimal_clusters(X_scaled):
    """Use the Elbow Method to determine the optimal number of clusters."""
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal Clusters')
    plt.savefig('images/elbow_method.png')
    plt.show()

def apply_kmeans(X_scaled, optimal_clusters=5):
    """Apply K-Means clustering and return cluster labels and model."""
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    return kmeans, clusters

def visualize_clusters(df, X, clusters, kmeans, scaler):
    """Visualize the clusters and save the plot."""
    df['Cluster'] = clusters
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'],
                    hue=df['Cluster'], palette='viridis', s=100)
    
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
    
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Customer Segmentation using K-Means')
    plt.legend()
    plt.savefig('images/customer_clusters.png')
    plt.show()

def main():
    """Main function to run K-Means clustering."""
    filepath = 'data/Mall_Customers.csv'  # Adjust if needed
    df = load_data(filepath)
    X, X_scaled, scaler = preprocess_data(df)
    find_optimal_clusters(X_scaled)
    kmeans, clusters = apply_kmeans(X_scaled, optimal_clusters=5)
    visualize_clusters(df, X, clusters, kmeans, scaler)
    df.to_csv('data/clustered_customers.csv', index=False)  # Save results

if __name__ == "__main__":
    main()
