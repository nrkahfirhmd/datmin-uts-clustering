import streamlit as st
import numpy as np
import pandas as pd
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

    image = resize(image, (out_h, int(out_h*float(w)/h)))
    
    st.image(image, caption="Input Image", width=750)

    if image.shape[-1] != 3:
        st.error("Image does not have expected RGB channels.")
        return

    data = np.array(image.reshape(-1, 3))

    clusters, pallete = kmeans(data, k=k)

    pallete_list = [tuple(color) for color in pallete]

    for color in pallete_list:
        color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        color_img[:, :] = (np.array(color) * 255).astype(np.uint8)

        color_pil_img = Image.fromarray(color_img)

        st.image(color_pil_img, width=100)
        st.write(to_hex(color))

    repaint(data, clusters, pallete_list, out_h)
    
    metrics(data, clusters)

def repaint(image, clusters, pallete, out_h=500):
    pic = image.copy()

    for px in range(image.shape[0]):
        pic[px] = pallete[clusters[px]]

    pic = pic.reshape(out_h, -1, 3)

    st.image(pic, caption="Output Image", width=750)

def metrics(data, label):
    silhouette = silhouette_score(data, label)
    davies_bouldin = davies_bouldin_score(data, label)
    
    st.text(f"Silhouette Coefficient : {round(silhouette, 4)}")
    st.text(f"Davies-Bouldin Index : {round(davies_bouldin, 4)}")