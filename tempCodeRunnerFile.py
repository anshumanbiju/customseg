from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import io

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
    try:
        # Load the data
        data = request.files['files']
        file_path = os.path.join(os.getcwd(), data.filename)
        data.save(file_path)
        print("Received data:", data)
        
        X = pd.DataFrame(data, columns=['Annual Income (k$)', 'Spending Score (1-100)'])

        # Perform DBSCAN clustering
        clusters = dbscan_clustering(X)
        print("Clusters:", clusters)

        # Visualize the clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=X, hue=clusters, palette='viridis')
        plt.title('DBSCAN Clustering')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend(title='Cluster')

        # Save the plot
        img_path = os.getcwd+'/cluster_plots.png'
        plt.savefig(img_path)
        # Converting Images to IOBytes
        img_bytes = io.StringIO()
        plt.savefig(img_bytes, format='svg')
        img_bytes.seek(0)

        # Converting to Context
        img_bytes = img_bytes.getvalue()
        context = {'images':img_bytes}
        
        plt.close()

        response = {
            'img_path': img_path
        }
        response = context
        return jsonify(response)
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
