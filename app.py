import streamlit as st
import tensorflow as tf
import cv2
import os
from PIL import Image, ImageOps
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
st.set_option('deprecation.showfileUploaderEncoding', False) # to avoid warnings while uploading files

# Here we will use st.cache so that we would load the model only once and store it in the cache memory which will avoid re-loading of model again and again.
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('my_model4.hdf5')
  return model

# load and store the model
with st.spinner('Model is being loaded..'):
  model=load_model()
  
# Function for prediction
def import_and_predict(image_data, model):
    size = (128,128)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction
def main():
    st.title("Monument Image Classifier")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Monument Image Classifier App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    file = st.file_uploader("Please upload an image", type=["jpg", "png"])
    class_names=['Angkor Wat','Buckingham Palace','Burj Khalifa','Christ the Redeemer','Gateway of India','Niagara Falls','Taj Mahal','The Eiffel Tower','The Great Wall of China','The Sydney Opera House']
    result=""
    final_images=""
    with st.sidebar:
      with st.expander("Upload an image of one of these monuments"):
        
#       st.header("Please upload an image from one of these categories")
        st.text("1. Angkor Wat")
        st.text("2. Buckingham Palace")
        st.text("3. Burj Khalifa")
        st.text("4. Christ the Redeemer")
        st.text("5. Gateway of India")
        st.text("6. Niagara Falls")
        st.text("7. Taj Mahal")
        st.text("8. The Eiffel Tower")
        st.text("9. The Great Wall of China")
        st.text("10. The Sydney Opera House")
      st.header("Monument Classifier using VGG16")
      st.image("vgg16.jpg")
            
    if st.button("Predict"):
      if file is None:
        st.write("please upload an image")
      else:
        image = Image.open(file)
        
        predictions = import_and_predict(image,model)
        score = tf.nn.softmax(predictions[0])
        result= class_names[np.argmax(predictions[0])]
#         st.write('This is {} '.format(result))
        html_temp = f"""
                    <div style="background-color:tomato;padding:10px">
                    <h2 style="color:white;text-align:center;"> This is {result} </h2>
                    </div>
                     """
        st.markdown(html_temp,unsafe_allow_html=True)
        st.image(image, use_column_width=True)
      
      
        st.caption("The result is trained on similar images like: ")
        
        train_path_Angkor_wat=[]
        train_path_Buckingham_Palace=[]
        train_path_Burj_khalifa=[]
        train_path_Christ_the_Redeemer=[]
        train_path_Gateway_of_India=[]
        train_path_Niagara_Falls=[]
        train_path_Tajmahal=[]
        train_path_The_Eiffel_Tower=[]
        train_path_The_Great_Wall_of_China=[]
        train_path_The_Sydney_Opera_House=[]
        
        for folder_name in ['Angkor_wat/','Buckingham_Palace/','Burj_khalifa/','Christ_the_Redeemer/','Gateway_of_India/','Niagara_Falls/','Tajmahal/','The_Eiffel_Tower/','The_Great_Wall_of_China/','The_Sydney_Opera_House/']:
    
          #Path of the folder
          images_path = os.listdir(folder_name)

          for i, image_name in enumerate(images_path): 
            if folder_name=='Angkor_wat/':
                train_path_Angkor_wat.append(folder_name+image_name)
            elif folder_name=='Buckingham_Palace/':
                train_path_Buckingham_Palace.append(folder_name+image_name)
            elif folder_name=='Burj_khalifa/':
                train_path_Burj_khalifa.append(folder_name+image_name)
            elif folder_name=='Christ_the_Redeemer/':
                train_path_Christ_the_Redeemer.append(folder_name+image_name)
            elif folder_name=='Gateway_of_India/':
                train_path_Gateway_of_India.append(folder_name+image_name)
            elif folder_name=='Niagara_Falls/':
                train_path_Niagara_Falls.append(folder_name+image_name)
            elif folder_name=='Tajmahal/':
                train_path_Tajmahal.append(folder_name+image_name)
            elif folder_name=='The_Eiffel_Tower/':
                train_path_The_Eiffel_Tower.append(folder_name+image_name)
            elif folder_name=='The_Great_Wall_of_China/':
                train_path_The_Great_Wall_of_China.append(folder_name+image_name)
            elif folder_name=='The_Sydney_Opera_House/':
                train_path_The_Sydney_Opera_House.append(folder_name+image_name)

        Angkor_wat=[]
        Buckingham_Palace=[]
        Burj_khalifa=[]
        Christ_the_Redeemer=[]
        Gateway_of_India=[]
        Niagara_Falls=[]
        Tajmahal=[]
        The_Eiffel_Tower=[]
        The_Great_Wall_of_China=[]
        The_Sydney_Opera_House=[]
        
        for i in train_path_Angkor_wat:
          image = Image.open(i).resize((200, 200))
          Angkor_wat.append(image)
        for i in train_path_Buckingham_Palace:
          image = Image.open(i).resize((200, 200))
          Buckingham_Palace.append(image)
        for i in train_path_Burj_khalifa:
          image = Image.open(i).resize((200, 200))
          Burj_khalifa.append(image)
        for i in train_path_Christ_the_Redeemer:
          image = Image.open(i).resize((200, 200))
          Christ_the_Redeemer.append(image)
        for i in train_path_Gateway_of_India:
          image = Image.open(i).resize((200, 200))
          Gateway_of_India.append(image)
        for i in train_path_Niagara_Falls:
          image = Image.open(i).resize((200, 200))
          Niagara_Falls.append(image)
        for i in train_path_Tajmahal:
          image = Image.open(i).resize((200, 200))
          Tajmahal.append(image)
        for i in train_path_The_Eiffel_Tower:
          image = Image.open(i).resize((200, 200))
          The_Eiffel_Tower.append(image)
        for i in train_path_The_Great_Wall_of_China:
          image = Image.open(i).resize((200, 200))
          The_Great_Wall_of_China.append(image)
        for i in train_path_The_Sydney_Opera_House:
          image = Image.open(i).resize((200, 200))
          The_Sydney_Opera_House.append(image)
        







        if result=='Angkor Wat':
            final_images =Angkor_wat

        elif result=='Buckingham Palace':
            final_images =Buckingham_Palace

        elif result=='Burj Khalifa':
            final_images =Burj_khalifa

        elif result=='Christ the Redeemer':
            final_images =Christ_the_Redeemer

        elif result=='Gateway of India':
            final_images =Gateway_of_India

        elif result=='Niagara Falls':
            final_images =Niagara_Falls

        elif result=='Taj Mahal':
            final_images =Tajmahal

        elif result=='The Eiffel Tower':
            final_images =The_Eiffel_Tower

        elif result=='The Great Wall of China':
            final_images =The_Great_Wall_of_China

        elif result=='The Sydney Opera House':
            final_images =The_Sydney_Opera_House

        n_rows = 1 + len(final_images) // int(4)
        rows = [st.container() for _ in range(n_rows)]
        cols_per_row = [r.columns(4) for r in rows]
        cols = [column for row in cols_per_row for column in row]

        for image_index, mon_image in enumerate(final_images):
            cols[image_index].image(mon_image)

    if st.button("About"):
     st.text("This is a monument classifier that predicts the monument of the image uploaded for 10 different monuments (list provided on the left).")
     st.text("This classifier uses VGG16, a pre-trained Convolutional Neural Network architecture.")
     st.text("This classifier has been deployed using Streamlit.")
if __name__=='__main__':
    main()
