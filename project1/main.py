import cv2
import numpy as np 
import streamlit as st
from tensorflow.keras.applications.mobile_net_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image

def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224,224))
    img = preprocess_input(image)
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(model, image):
    try:
        preprocess_image = preprocess_image(image)
        predictions =  model.predict(preprocess_image)
        decoded_predictions = decoded_predictions(predictions, top=3)[0]
        return decoded_predictions
   
    except Exception as e:
        st.error(f"Error Classifying Image: {str(e)}")
        return None
    
def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️", layout="centered")
    st.title("AI Image Classifier")
    st.write("Upload an image and let the AI tell you what is in it!")
    @st.cache_resource
    def load_cache_model():
        return load_model()
    
    model = load_cache_model()
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is None:
        image = st.image(
            uploaded_file, caption="Uploaded Image", use_container_width=True
        )
        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Analyzing Image..."):
                image = image.open(uploaded_file)
                predictions = classify_image(image)

                if predictions:
                    st.subheader("Predictions")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")
