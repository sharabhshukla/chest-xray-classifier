# This is a sample Python script that runs the chest x-ray scanning app
import streamlit as st
from PIL import Image, ImageOps
from keras.models import load_model
from loguru import logger
import numpy as np


def load_ml_model():
    keras_model = load_model('./ml_model/keras_model.h5')
    return keras_model


st.title("Pneumonia X-Ray scanning app")
st.text("This app used a ML model trained on thousands on chest x-rays for identifying "
        "pneumonia, it has a reasonable accuracy north of 90%. ")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
keras_model = load_ml_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
with st.form(key='classification form'):
        submit_button = st.form_submit_button(label='submit')
        if submit_button:
            st.write("Classifying...")
            if keras_model is not None:
                keras_model = load_ml_model()
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                # Replace this with the path to your image
                # resize the image to a 224x224 with the same strategy as in TM2:
                # resizing the image to be at least 224x224 and then cropping from the center
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.ANTIALIAS)
                # turn the image into a numpy array
                image_array = np.asarray(image)
                # Normalize the image
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                # Load the image into the array
                data[0] = normalized_image_array
                prediction = keras_model.predict(data)
                logger.info('prediction -> {}'.format(prediction))
                label_dict = {'0': 'Normal', '1': 'Pneumonia'}
                logger.info(label_dict)
                normal_prob = round(prediction[0][0],4)
                pneumonia_prob = round(prediction[0][1],4)
                final_label = 'Normal' if normal_prob > pneumonia_prob else 'Pneumonia'
                final_proba = normal_prob if normal_prob > pneumonia_prob else pneumonia_prob
                st.write('{} with {:.2f}% confidence'.format(final_label, final_proba*100))

