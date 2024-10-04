import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.decomposition import PCA
import pickle

centroids = pd.read_csv('centroids.csv')
centroids = np.array(centroids)
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
with open('../pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)

def preprocess_image(img_raw, target_size): 
    img = image.load_img(img_raw, target_size=target_size)   
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def assign_clusters(data, centroids):
    clusters = []
    
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        clusters.append(closest_centroid)
        
    return np.array(clusters)


def predict(image):
    image_size = (224, 224)
    
    img = preprocess_image(image, image_size)
    features = model.predict(img)  # Extract features using VGG16
    features_flattened = features.flatten().reshape(1, -1)  # Flatten features
        
    pca_features = pca.transform(features_flattened)
    
    cluster = assign_clusters(pca_features, centroids)        
    
    return cluster