import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json

def precompute_clusters():
    # Load the cleaned data
    df = pd.read_csv('df_cleaned.csv')
    
    # Features for clustering
    features = [
        'temperature', 'apparentTemperature', 'dewPoint', 'humidity', 
        'pressure', 'precipIntensity', 'ozone', 'precipAccumulation',
        'hour', 'day_of_week', 'month'
    ]
    
    # Get features for all cities combined
    X = df[features]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute PCA once for all data
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X_scaled)
    
    # Initialize results dictionary
    results = {
        "pca_coords": pca_coords.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist()
    }
    
    # Compute clusters for k values from 1 to 10
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)  # Use original scaled features
        results[f"k{k}_labels"] = labels.tolist()
        
        # Store cluster centers in PCA space
        centers_pca = pca.transform(kmeans.cluster_centers_)
        results[f"k{k}_centers"] = centers_pca.tolist()
        
        # Store inertia (within-cluster sum of squares)
        results[f"k{k}_inertia"] = float(kmeans.inertia_)
    
    # Save results to JSON file
    with open('cluster_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    precompute_clusters()