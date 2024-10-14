import streamlit as st
import numpy as np
import pandas as pd
from skimage.transform import resize
from imageio.v2 import imread
from kmeans import kmeans
from matplotlib.colors import to_hex
from PIL import Image

def predict_color(file, k):
    image = imread(file)

    h, w = image.shape[:2]
    out_h = 500

    image = resize(image, (out_h, int(out_h*float(w)/h)), anti_aliasing=True)
    
    st.image(image, caption="Input Image")

    if image.shape[-1] != 3:
        st.error("Image does not have expected RGB channels.")
        return

    data = np.array(image.reshape(-1, 3))

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

    repaint(data, clusters, pallete_list)

def repaint(image, clusters, pallete, out_h=500):
    pic = image.copy()

    for px in range(image.shape[0]):
        pic[px] = pallete[clusters[px]]

    pic = pic.reshape(out_h, -1, 3)

    st.image(pic, caption="Output Image")
