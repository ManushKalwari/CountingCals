# -*- coding: utf-8 -*-
"""App.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lWwGBGB1LWholtlqDyi1W0P3QfS0STTm
"""

#!pip install streamlit
!pip install tensorflow

import streamlit as st # framework for datascience web based applications
from PIL import Image
import requests
from tensorflow.keras.preprocessing.image import load_img,img_to_array

from keras.models import load_model
model = load_model('/content/CountingCals.h5')

labels = {0:'Apple Red', 1:'Apple Red Delicious', 2:'Apple Red Yellow', 3:'Bread White', 4:'Banana Cavendish', 5:'Boiled Eggs', 6:'Apple Braeburn',
          7:'Chapati', 8:'Banana Red', 9:'Apple Golden', 10:'Naan', 11:'Akki Roti', 12:'Rice Masuri', 13:'Ragi Roti', 14:'Apple Granny Smith', 
          15:'Bread Brown', 16: 'Banana Elaichi', 18: 'Rice Basmati', 19:'Rice Mogara', 20:'Thalipeeth'}

import numpy as np
import pandas as pd

df = pd.read_csv('calories.csv')

page_style = """
<style>
[data-testid = "stAppViewContainer"] {
background: #006eb3;
box-shadow: inset 19px 19px 38px #003f66,
            inset -19px -19px 38px #009dff;
}

[data-testid = "stToolbar"] {
border-radius: 12px;
background: #006eb3;
box-shadow: inset 11px 11px 21px #003f66,
            inset -11px -11px 21px #009dff;
}

[data-testid = "stHeader"] {
background: #006eb3;
}

[data-testid = "stFileUploadDropzone"] {
border-radius: 18px;
background: #006eb3;
box-shadow: inset 11px 11px 21px #003f66,
            inset -11px -11px 21px #009dff;
}

[data-testid = "stImage"] img{
border-radius: 20%;
}

[data-testid = "column"] {
border-radius: 18px;
background: #006eb3;
padding:16px;
box-shadow: inset 11px 11px 21px #003f66,
            inset -11px -11px 21px #009dff;
}

[id = "countingcals"] {
text-align: center;
color: #143628;
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
    st.title("CountingCals🍍")
    st.subheader("Just Upload a photo of your serving. Get to know the calories you consume.")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = '/content/Image_13.png'
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # if st.button("Predict"):
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
                calorie_std = str(calorie_value) + " / " + str(calorie_std)




            #st.text("Image succesfully uploaded {}".format(result))
            st.text("Image uploaded successfully")
            col1,col2 = st.columns(2)
            col1.metric("Food item:",result)
            col2.metric("Calorie:",calorie_std)
            print(resp)

run()
