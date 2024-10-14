import streamlit as st
import numpy as np
import pandas as pd
from skimage.transform import resize
from imageio.v2 import imread
from kmeans import kmeans
from sklearn.cluster import KMeans
from matplotlib.colors import to_hex
from PIL import Image

def predict_color(file, k):
    image = imread(file)
    
    st.image(image, caption="Input Image")
    
    image = resize(image, (250, 250), anti_aliasing=True)
    
    if image.shape[-1] != 3:
        st.error("Image does not have expected RGB channels.")
        return
    
    data = image.reshape(-1, 3)
    
    clusters, pallete = kmeans(data, k=k)
    
    pallete_list = list()
    
    for color in pallete:
        pallete_list.append(tuple(color))
    
    for color in pallete_list:
        color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        color_img[:, :] = (np.array(color) * 255).astype(np.uint8)

        color_pil_img = Image.fromarray(color_img)

        st.image(color_pil_img, width=100)
        st.write(to_hex(color))