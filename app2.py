import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO
st.set_option('deprecation.showfileUploaderEncoding', False) # to avoid warnings while uploading files
st.title("Monument Image Classifier")
# Here we will use st.cache so that we would load the model only once and store it in the cache memory which will avoid re-loading of model again and again.
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('my_model4.hdf5')
  return model

# load and store the model
with st.spinner('Model is being loaded..'):
  model=load_model()


# upload the image
file = st.file_uploader("Please upload an image", type=["jpg", "png"])
class_names=['Angkor_wat','Buckingham_Palace','Burj_khalifa','Christ_the_Redeemer','Gateway_of_India','Niagara_Falls','Tajmahal','The_Eiffel_Tower','The_Great_Wall_of_China','The_Sydney_Opera_House']




# Function for prediction
def import_and_predict(image_data, model):

        size = (128,128)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

        img_reshape = img[np.newaxis,...]

        prediction = model.predict(img_reshape)

        return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image,model)
    score = tf.nn.softmax(predictions[0])
    st.write(predictions)
    # st.write(score)
    st.write(
    "This image most likely belongs to {}."
    .format(class_names[np.argmax(score)])
)
