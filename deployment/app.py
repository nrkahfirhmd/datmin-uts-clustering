import streamlit as st
from functions import predict
from PIL import Image
import numpy as np
import io

st.title('Guessing Your Image')

image = st.file_uploader("Choose an Image", type=["png", "jpg", "jpeg"])

if image is not None:
    img_bytes = image.read()
    
    img_file = io.BytesIO(img_bytes)    
    
    result = predict(img_file)

    labels = ['Looka Like a Residential', 'Looks Like an Airport', 'Looks Like A Stadium', 'Looks Like a Mountain or Forest', 'Looks Like a Farmland', 'Looks Like A Beach or Desert', 'Looks Like a Stadium', 'Looks Like A Residential']
    
    st.image(img_file)
    st.header(labels[result[0]])
    # st.image(img)
