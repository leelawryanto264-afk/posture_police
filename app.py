import streamlit as st
import numpy as np
import tensorflow as tf
import PIL as p
from teachable_machine import TeachableMachine

st.title("Posture Police")
st.markdown("desc do later")

@st.cache_resource
def pos_model():
    model = TeachableMachine(model_path="keras_model.h5", labels_file_path="labels.txt")
    return model
cam = pos_model()

# add a help setion later
st.sidebar.header("Settings")
angle = st.sidebar.slider("Slouch Angle", 120, 160, 140)
conf = st.sidebar.slider("Sensitivity", 0.1, 0.9, 0.3)

img_temp = st.camera_input("Take a picture")

if img_temp is not None:
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(img_temp.getvalue())
    result = cam.classify_image(temp_path)
    class_name = result["class_name"]
    class_sens = result["class_sens"]

    st.write(f"Pose is: {class_name}")
    st.write(f"Model is Certain by: {int(class_sens)}")