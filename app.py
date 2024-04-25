import os
import io
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

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

def preprocess_data(df):
    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Convert 'Gender' to numerical values

    # Exclude non-numeric columns and unnecessary columns
    numeric_df = df.drop(['CustomerID'], axis=1)  # Exclude 'CustomerID'

    return numeric_df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/introduction')
def introduction():
    return render_template('introduction.html')

@app.route('/segmentation', methods=['GET', 'POST'])
def segmentation():
    if request.method == 'POST':
        # Get the uploaded dataset from the request
        uploaded_dataset = request.files.get('dataset')

        # Read the uploaded dataset
        if uploaded_dataset:
            df = pd.read_csv(uploaded_dataset)
        else:
            df = None

        # Perform clustering and get necessary data
        silhouette_score_val, num_clusters, cluster_details, img_path = None, None, {}, None
        if df is not None:
            # Preprocess the data
            X = preprocess_data(df)
            
            # Perform DBSCAN clustering
            clusters = dbscan_clustering(X)

            # Calculate silhouette score
            silhouette_score_val = silhouette_score(X, clusters)

            # Get number of clusters
            num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

            # Get cluster details
            unique_clusters = np.unique(clusters)
            for cluster_label in unique_clusters:
                if cluster_label != -1:  # Ignore noise points
                    cluster_points = X[clusters == cluster_label]
                    cluster_center = cluster_points.mean(axis=0)
                    cluster_size = len(cluster_points)
                    cluster_details[str(cluster_label)] = {  # Convert cluster_label to string
                        'center': cluster_center.tolist(),
                        'size': cluster_size
                    }

            # Visualize the clusters
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=X, hue=clusters, palette='viridis')
            plt.title('DBSCAN Clustering')
            plt.xlabel('Annual Income (k$)')
            plt.ylabel('Spending Score (1-100)')
            plt.legend(title='Cluster')

            # Save the plot in the static folder
            img_path = os.path.join(app.static_folder, 'cluster_plots.png')
            plt.savefig(img_path)
            plt.close()

        # Pass the dataset and clustering details to the template and render segmentation.html
        return render_template('segmentation.html', dataset=df, silhouette_score=silhouette_score_val, num_clusters=num_clusters, cluster_details=cluster_details, img_path=img_path)
    else:
        # Render segmentation.html without dataset if it's a GET request
        return render_template('segmentation.html', dataset=None, silhouette_score=None, num_clusters=None, cluster_details={}, img_path=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:    
        # Load the data
        data = request.files['files']
        X = pd.read_csv(data)

        # Encode categorical variables like 'Gender' using one-hot encoding
        X = pd.get_dummies(X, columns=['Gender'])

        # Extract file name and column names
        filename = data.filename
        column_names = X.columns.tolist()

        # Perform DBSCAN clustering
        clusters = dbscan_clustering(X.drop(columns=['CustomerID']))  # Exclude 'CustomerID'

        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, clusters)

        # Get number of clusters
        num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

        # Get cluster details
        cluster_details = {}
        unique_clusters = np.unique(clusters)
        for cluster_label in unique_clusters:
            if cluster_label != -1:  # Ignore noise points
                cluster_points = X[clusters == cluster_label]
                cluster_center = cluster_points.mean(axis=0)
                cluster_size = len(cluster_points)
                cluster_details[str(cluster_label)] = {  # Convert cluster_label to string
                    'center': cluster_center.tolist(),
                    'size': cluster_size
                }

        # Visualize the clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=X, hue=clusters, palette='viridis')
        plt.title('DBSCAN Clustering')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend(title='Cluster')

        # Save the plot in the static folder
        img_path = os.path.join(app.static_folder, 'cluster_plots.png')
        plt.savefig(img_path)

        plt.close()

        # Prepare dataset object to pass to frontend
        dataset = {
            'filename': filename,
            'column_names': column_names,
            'values': X.values.tolist()  # Convert dataframe to list of lists
        }

        response = {
            'img_path': '/static/cluster_plots.png',  # Return the path relative to the static folder
            'dataset': dataset,  # Pass the dataset object
            'silhouette_score': silhouette_avg,
            'num_clusters': num_clusters,
            'cluster_details': cluster_details  # Pass the cluster details
        }
        return jsonify(response)
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
