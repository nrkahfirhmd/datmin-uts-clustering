import streamlit as st 
import functions as f 

st.title("Image Segmentation Clustering")

file = st.file_uploader("Upload a JPEG, JPG, or PNG Image", ['.JPEG', ".JPG", ".PNG"])
k = st.number_input("Numbers of Clusters", min_value=2, max_value=10, value=5)
predict = st.button("Predict")

if predict and file is not None:
    f.predict_color(file, k)