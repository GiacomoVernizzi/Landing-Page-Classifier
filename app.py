import pandas as pd
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

st.title('Landing Page Classfier')

st.subheader('Unsure whether your landing page will convert or not? Check this out!')

st.caption('This is a CNN classifier trained on 100 good and 100 bad Ad landing pages. \
Just screeshot your landing page and drop it into the box below to generate an instant prediction!', unsafe_allow_html=False)

model = tf.keras.models.load_model('LP_classifier.h5')
### load file
uploaded_file = st.file_uploader("Choose a image file", type=['png', 'jpg', 'jpeg'])


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(256,256))
    # Display image:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]


    Genrate_pred = st.button("Generate Prediction")    
    
    
    if Genrate_pred:
        prediction = model.predict(img_reshape)
        if prediction < 0.25:
            print(st.subheader("This landing page is very likely to convert"))
        elif 0.26 > prediction < 0.5:
            print(st.subheader("This landing page is likely to convert"))
        elif 0.5 > prediction < 0.75:
            print(st.subheader("This landing page is unlikely to convert"))
        elif prediction > 0.75:
            print(st.subheader("This landing page is very unlikely to convert"))
            

   
