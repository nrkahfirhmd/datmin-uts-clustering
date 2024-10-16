import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from imageio.v2 import imread
from kmeans import kmeans
from matplotlib.colors import to_hex
from PIL import Image
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

def predict_color(file, k):
    image = imread(file)

    h, w = image.shape[:2]
    out_h = 250

    st.image(image, caption="Input Image", width=750)

    image = resize(image, (out_h, int(out_h*float(w)/h)), anti_aliasing=True)

    if image.shape[-1] != 3:
        st.error("Image does not have expected RGB channels.")
        return

    data = np.array(image.reshape(-1, 3))

    clusters, pallete = kmeans(data, k=k)
    
    pallete_list = [tuple(color) for color in pallete]
    
    st.header("Feature Extractions")

    st.subheader("Dominant Color")
    get_dominant_colors(pallete_list, k)
    
    st.subheader("Color Distributions")
    get_color_distribution(clusters, data.shape[0], pallete_list, k)

    st.header("Clustering Result")
    metrics(data, clusters)
    
    repaint(data, clusters, pallete_list, out_h)

def repaint(image, clusters, pallete, out_h=500):
    pic = image.copy()

    for px in range(image.shape[0]):
        pic[px] = pallete[clusters[px]]

    pic = pic.reshape(out_h, -1, 3)

    st.image(pic, caption="Output Image", width=750)

def metrics(data, label):
    silhouette = silhouette_score(data, label)
    davies_bouldin = davies_bouldin_score(data, label)
    
    metrics1, metrics2 = st.columns(2)
    
    metrics1.metric("Silhouette Coefficient", round(silhouette, 4))
    metrics2.metric("Davies-Bouldin Index", round(davies_bouldin, 4))
    
def get_dominant_colors(pallete_list, k):
    cols = st.columns(k)
    
    for i, color in enumerate(pallete_list):
        color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        color_img[:, :] = (np.array(color) * 255).astype(np.uint8)

        color_pil_img = Image.fromarray(color_img)

        with cols[i]:
            st.image(color_pil_img, width=100)
            st.write(to_hex(color))
            
def get_color_distribution(clusters, pixels, colors, k):
    percentage = np.asarray(np.unique(clusters, return_counts=True)[1], dtype='float32')
    percentage = percentage / pixels
    
    dominance = [[percentage[px], colors[px]] for px in range(k)]
    dominance = sorted(dominance, key=lambda x: x[0], reverse=True)
    
    patch = np.zeros((50, 750, 3), dtype=np.uint8)
    
    start = 0
    for cx in range(k):
        width = int(dominance[cx][0] * patch.shape[1])
        end = start + width
        patch[:, start:end, :] = (np.array(dominance[cx][1]) * 255).astype(np.uint8)
        start = end
    
    st.image(Image.fromarray(patch))