from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

def dbscan_clustering(X):
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply DBSCAN clustering
    eps = 0.4  # distance epsilon
    min_samples = 4
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)

    return clusters

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the data
    data = request.get_json()
    X = pd.DataFrame(data, columns=['Annual Income (k$)', 'Spending Score (1-100)'])

    # Perform DBSCAN clustering
    clusters = dbscan_clustering(X)

    # Visualize the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=X, hue=clusters, palette='viridis')
    plt.title('DBSCAN Clustering')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend(title='Cluster')

    # Save the plot
    img_path = 'static/cluster_plots.png'
    plt.savefig(img_path)
    plt.close()

    response = {
        'img_path': img_path
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
