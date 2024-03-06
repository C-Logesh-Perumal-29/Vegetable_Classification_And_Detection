# Importing Libraries :
import streamlit as st
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os
import h5py
import requests    
import json
import streamlit as st
import streamlit.components.v1 as com
import cv2
import numpy as np
from PIL import Image
import cvzone
import math
from ultralytics import YOLO


# Setting Icon Image :
img = Image.open("Icon.png")
st.set_page_config(page_title="Vegetable Classification & Detection",page_icon=img,layout="wide")

# Hide Menu_Bar & Footer :
hide_menu_style = """
    <style>
    #MainMenu {visibility : hidden;}
    footer {visibility : hidden;}
    </style>
"""
st.markdown(hide_menu_style , unsafe_allow_html=True)

# Set the background image :
Background_image = """
<style>
[data-testid="stAppViewContainer"] > .main
{
background-image: url("https://img.freepik.com/free-vector/abstract-watercolor-design_1055-7990.jpg?w=740&t=st=1709652911~exp=1709653511~hmac=d7d05c8a11f19a9bf91f7e4534342dbabf0aa64dbad0b89e51a4fdc8ce4e8f68");
background-size : 100%
background-position : top left;
background-position: center;
background-size: cover;
background-repeat : repeat;
background-repeat: round;
background-attachment : local;

background-image: url("https://img.freepik.com/free-vector/abstract-watercolor-design_1055-7990.jpg?w=740&t=st=1709652911~exp=1709653511~hmac=d7d05c8a11f19a9bf91f7e4534342dbabf0aa64dbad0b89e51a4fdc8ce4e8f68");
background-position: right bottom;
background-repeat: no-repeat;
}  
[data-testid="stHeader"]
{
background-color : rgba(0,0,0,0);
}
</style>                                
"""
st.markdown(Background_image,unsafe_allow_html=True)

title_html = f'<h1 style="color:#219ebc; text-align:center;font-family:Edwardian Script ITC;font-size:96px;">Vegetable Classification & Detection</h1>'
st.markdown(title_html, unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col2:
    st.image("V1.png")


# Marquee Tag - About VC :
st.markdown("""
    <marquee width="100%" direction="left" height="100px" scrollamount="6" style="color:black;font-family:Maiandra GD;">
    * Vegetable classification is a crucial process in Agriculture and Food Processing. * Supervised Learning technique is used to train the model. * Convolutional Neural Network (CNN) is a deep learning technology that is mainly used in Image Recognition and Classification tasks. * Image Classification is the process of categorizing an image into a predefined classes based on the tasks. * The model will learn based on the pixels and the model will recognize the pattern and extract the feature and training with the predefined classes/labels. * Vegetables Detection employs the YOLO (You Only Look Once) model for object detection, which identifies vegetables in the live feed with bounding boxes and labels.
    </marquee>
""",unsafe_allow_html=True)

# Creating Columns for Audio and Image :
col1,col2,col3,col4 = st.columns([2,4,4,4])
with col1:
    st.markdown("""
                <h6 style="color:#4D160D;font-family:Hobo Std;"> Audio File </h6>
                """,unsafe_allow_html=True)          
    btn = st.button("Click")
    if btn:
        st.audio("VC.mp3")
with col2:
    st.markdown("""
                <marquee width="100%" direction="up" height="150px" scrollamount="6">
                <img src="https://img.freepik.com/free-photo/top-view-ripe-fresh-tomatoes-with-water-drops-black-background_141793-3432.jpg?size=626&ext=jpg&ga=GA1.1.2087154549.1663432512&semt=ais" alt="VC Image 1">
                </marquee>
                """,unsafe_allow_html=True)
with col3:
    st.markdown("""
                <marquee width="100%" direction="up" height="150px" scrollamount="6">
                <img src="https://img.freepik.com/free-photo/corn-texture_1308-4992.jpg?size=626&ext=jpg&ga=GA1.1.2087154549.1663432512&semt=sph" alt="VC Image 2">
                </marquee>
                """,unsafe_allow_html=True)
with col4:
    st.markdown("""
                <marquee width="100%" direction="up" height="150px" scrollamount="6">
                <img src="https://img.freepik.com/free-photo/picture-vegetables-table_1340-24016.jpg?size=626&ext=jpg&ga=GA1.1.2087154549.1663432512&semt=sph" alt="VC Image 3">
                </marquee>
                """,unsafe_allow_html=True)
               
m = st.markdown(""" <style> div.stRadio > checkbox { 
                    color : black; } 
                    </style>""", unsafe_allow_html=True)
                    
#  Main Part :

Options = st.selectbox("Select your choice",["Select Your Choice","Vegetable Classification","Vegetable Detection","Predictions"])

if Options == "Vegetable Classification":

    col_1,col_2 = st.columns([5,5])
    
    with col_1:    
        
            def main():
                file_uploader = st.file_uploader("Choose the file",type = ['jpg','jpeg','png'])
                if file_uploader is not None:
                    image = Image.open(file_uploader)
                    figure = plt.figure()
                    plt.imshow(image)
                    plt.axis("off")
                    result = predict_class(image)
                    st.pyplot(figure)
                    st.success(result) 
                                             
            def predict_class(image):       
                classifier_model = tf.keras.models.load_model("D:\\Project_Web\\Updated Vegetable Classification and Detection\\VC Model.h5")
                shape = ((228,228,3))
                tf.keras.Sequential([hub.KerasLayer(classifier_model,input_shape=shape)])
                test_image = image.resize((228,228))
                test_image = preprocessing.image.img_to_array(test_image)
                test_image = test_image/255.0
                test_image = np.expand_dims(test_image,axis=0)
                class_names = ['Beans', 'Beetroot', 'Bittergourd', 'Brinjal', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Drumstick', 'Garlic', 'Ginger', "Lady's finger", 'Lemon', 'Onion', 'Peas', 'Potato', 'Pumpkin', 'Radish', 'Tomato'] 
                predictions = classifier_model.predict(test_image)
                scores = tf.nn.softmax(predictions[0])
                scores = scores.numpy()
                image_class = class_names[np.argmax(scores)]
                result = "Prediction of the uploaded image is : {}".format(image_class)
                return result
                
                st.balloons()
            
            if __name__ == "__main__":
                main()
    
            st.balloons()
    
    with col_2:
            
            up = st.camera_input("Capture image",help="This is just a basic example")
    
            filename = up.name
            with open(filename,'wb') as imagefile:
                imagefile.write(up.getbuffer())
    
            def main():
                
                if up is not None:
                    image = Image.open(up)
                    figure = plt.figure()
                    plt.imshow(image)
                    plt.axis("off")
                    result = predict_class(image)
                    st.pyplot(figure)
                    st.success(result)
                    
            def predict_class(image):
                classifier_model = tf.keras.models.load_model("D:\\Project_Web\\Updated Vegetable Classification and Detection\\VC Model.h5")
                shape = ((228,228,3))
                tf.keras.Sequential([hub.KerasLayer(classifier_model,input_shape=shape)])
                test_image = image.resize((228,228))
                test_image = preprocessing.image.img_to_array(test_image)
                test_image = test_image/255.0
                test_image = np.expand_dims(test_image,axis=0)
                class_names = ['Beans', 'Beetroot', 'Bittergourd', 'Brinjal', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Drumstick', 'Garlic', 'Ginger', "Lady's finger", 'Lemon', 'Onion', 'Peas', 'Potato', 'Pumpkin', 'Radish', 'Tomato']
                predictions = classifier_model.predict(test_image)
                scores = tf.nn.softmax(predictions[0])
                scores = scores.numpy()
                image_class = class_names[np.argmax(scores)]
                result = "Prediction of the captured image is : {}".format(image_class)
                return result
            
                st.balloons()
    
            if __name__ == "__main__":
                main()
            
            st.balloons()
            
elif Options == "Vegetable Detection":
    
    # Function to perform object detection and draw bounding boxes
    def perform_object_detection(image, model, classNames):
        results = model(image, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # Convert coordinates to integers
                x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)  # Ensure non-negative coordinates
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), thickness=3) # Specify thickness explicitly
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(image, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                cvzone.putTextRect(image, f'{classNames[cls]} {conf}', (max(1, x1), max(0, y1))) # Adjusted y coordinate
        return image

    # Load YOLO model
    model = YOLO("best.pt")

    # Define class names
    classNames = ['Beans', 'Beetroot', 'Bitterguard', 'Brinjal', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower',
                  'Corn', 'Garlic', 'Ginger', 'Green Chilli', 'Ladys Finger', 'Lemon', 'Onion', 'Peas', 'Potato',
                  'Radish', 'Red Chilli', 'Tomato']


    # Webcam capture function
    def capture_image(cap, is_detection_started):
        while is_detection_started:
            ret, frame = cap.read()
            if not ret:
                break

            frame = perform_object_detection(frame, model, classNames)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame

    # Main Streamlit code
    if __name__ == "__main__":
        cap = cv2.VideoCapture(0)
        cap.set(3, 1500)
        cap.set(4, 720)

        is_detection_started = False
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            start_button = st.button("Start")
        with col2:
            stop_button = st.button("Stop")

        if start_button:
            is_detection_started = True

        if stop_button:
            is_detection_started = False

        if is_detection_started:
            frames = capture_image(cap, is_detection_started)
            frame = next(frames)

            stframe = st.empty()
            stframe.image(frame, channels="RGB")

            for frame in frames:
                stframe.image(frame, channels="RGB", use_column_width=True)

        cap.release()

        
elif Options == "Predictions":       
    
    b1,b2,b3,b4 = st.columns(4)
    with b1: 
        st.image("./Predictions/1 (1).png")
    with b2: 
        st.image("./Predictions/1 (2).png")  
    with b3: 
        st.image("./Predictions/1 (3).png")
    with b4:
        st.image("./Predictions/1 (4).png")
        
    b11,b22,b33,b44 = st.columns(4)
    with b11: 
        st.image("./Predictions/1 (5).png")
    with b22: 
        st.image("./Predictions/1 (6).png")
    with b33: 
        st.image("./Predictions/1 (7).png")
    with b44:
        st.image("./Predictions/1 (8).png")
        
    b111,b222,b333,b444 = st.columns(4)
    with b111: 
        st.image("./Predictions/1 (9).png")
    with b222: 
        st.image("./Predictions/1 (10).png")
    with b333: 
        st.image("./Predictions/1 (11).png")
    with b444:
         st.image("./Predictions/1 (12).png")
        
    b1111,b2222,b3333,b4444 = st.columns(4)
    with b1111: 
        st.image("./Predictions/1 (13).png")        
    with b2222: 
        st.image("./Predictions/1 (14).png")
    with b3333: 
         st.image("./Predictions/1 (15).png")
    with b4444: 
         st.image("./Predictions/1 (16).png")
              
    b11111,b22222,b33333,b44444 = st.columns(4)
    with b11111: 
        st.image("./Predictions/1 (17).png")
    with b22222: 
        st.image("./Predictions/1 (18).png")
    with b33333: 
        st.image("./Predictions/1 (19).png")
    with b44444: 
        st.image("./Predictions/1 (20).png")
        
    st.balloons()
        
        