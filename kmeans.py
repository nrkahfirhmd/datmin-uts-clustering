import numpy as np

# Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Initialize Centroid
def initialize_centroids(data, k):
    np.random.seed(42)
    random_indices = np.random.permutation(data.shape[0])
    centroids = data[random_indices[:k]]
    return centroids

# Clustering
def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        clusters.append(closest_centroid)
    return np.array(clusters)

# Update centroids
def update_centroids(data, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:            
            new_centroids.append(data[np.random.choice(data.shape[0])])
    return np.array(new_centroids)

# K-Means
def kmeans(data, k, max_iters=100, tol=1e-4):
    # Step 1: Initialize centroids
    centroids = initialize_centroids(data, k)
    
    for i in range(max_iters):
        # Step 2: Assign clusters
        clusters = assign_clusters(data, centroids)
        
        # Step 3: Update centroids
        new_centroids = update_centroids(data, clusters, k)
        
        # Step 4: Check for convergence (if centroids don't change)
        if np.all(np.abs(new_centroids - centroids) < tol):            
            break
        
        centroids = new_centroids

    return clusters, centroids
