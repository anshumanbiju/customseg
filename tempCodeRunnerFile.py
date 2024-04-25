def predict():
    try:    
        # Load the data
        data = request.files['files']
        X = pd.read_csv(data)

        # Extract file name and column names
        filename = data.filename
        column_names = X.columns.tolist()

        # Perform DBSCAN clustering
        clusters = dbscan_clustering(X[['Annual Income (k$)', 'Spending Score (1-100)']])

        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, clusters)

        # Get number of clusters
        num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

        # Get cluster details
        cluster_details = {}
        for cluster_label in set(clusters):
            if cluster_label != -1:  # Ignore noise points
                cluster_points = X[clusters == cluster_label]
                cluster_center = cluster_points.mean(axis=0)
                cluster_size = len(cluster_points)
                cluster_details[cluster_label] = {
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
            'cluster_details': cluster_details
        }
        return jsonify(response)
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500
