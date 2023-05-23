# -*- coding: utf-8 -*-
"""App.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lWwGBGB1LWholtlqDyi1W0P3QfS0STTm
"""

#!pip install streamlit
#pip tensorflow
import streamlit as st # framework for datascience web based applications
from PIL import Image
import requests
from tensorflow.keras.utils import load_img,img_to_array
import VolumeEstimation

from keras.models import load_model
model = load_model('CountingCals.h5')

labels = {0:'Bread White', 1:'Bread Brown', 2:'Boiled Egg', 3:'Rice', 4:'Banana Elaichi', 5:'Banana Cavendish', 6:'Apple'}

import numpy as np
import pandas as pd

df = pd.read_csv('calories.csv')

page_style = """
<style>

[data-testid = "stAppViewContainer"] {
background: #f0b000;
box-shadow: inset 41px 41px 63px #c08d00,
            inset -41px -41px 63px #ffd300;
}

[data-testid = "stToolbar"] {
border-radius: 12px;
background: #f0b000;
box-shadow: inset 11px 11px 17px #c59000,
            inset -11px -11px 17px #ffd000;
}

[data-testid = "stHeader"] {
background: #f0b000;
}

[data-testid = "stFileUploadDropzone"] {
border-radius: 18px;
background: #f0b000;
box-shadow: inset 11px 11px 17px #c59000,
            inset -11px -11px 17px #ffd000;
}

[data-testid = "stImage"] img{
border-radius: 20%;
}

[data-testid = "column"] {
border-radius: 18px;
background: #f0b000;
padding:16px;
box-shadow: inset 11px 11px 17px #c59000,
            inset -11px -11px 17px #ffd000;
}

[id = "countingcals"] {
text-align: center;
color: #00000;
font-size: 60px;
}

    

</style>
"""

st.markdown(page_style, unsafe_allow_html=True)


def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()

def run():
    #st.title("CountingCals🍍")
    new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">CountingCals🍍</p>'
    st.title(new_title)
    st.subheader("Upload photos of your food & know the calories you consume.")
    img_files = st.file_uploader("Choose Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    volume = getVolume(img_files)

    
            
    if img_files:
        displayed_images = 0  
        for img_file in img_files:
            img = Image.open(img_file).resize((150, 150))
            st.image(img, use_column_width=False)
            save_image_path = f"Image_{img_files.index(img_file)}.png"
            img.save(save_image_path)
            displayed_images += 1  
            if displayed_images == 3: 
                break
            
     

        if img_file is not None:
            result = processed_img(save_image_path)
            url = 'https://orange-paws-lose-34-125-199-204.loca.lt/'
            form_data = {'file': open(save_image_path, 'rb')}
            resp = requests.post(url, files=form_data)

            calorie_value = 0
            food_weight = 0
            calorie_std = 0

            for i in range(len(df)):
              if((df.iloc[i, 0]).lower() == result.lower()):
                calorie_value = df.iloc[i, 1]
                food_weight = df.iloc[i, 2]
                calorie_std = str(calorie_value) + " / " + str(food_weight)

            st.text("Images uploaded successfully")
            col1,col2 = st.columns(2)
            col1.metric("Food item:",result)
            col2.metric("Calorie:",calorie_std)
            print(resp)

run()
