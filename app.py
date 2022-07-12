from unittest import result
import google_trans_new_main.google_trans_new1
from google_trans_new_main.google_trans_new1 import google_translator
import tensorflow as tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from numpy import asarray
from PIL import Image
import streamlit as st
import os
import cv2
import joblib
import streamlit.components.v1 as components
from gtts import gTTS
import datetime as dt
import speech_recognition as sr
from pathlib import Path

#voice recorder
parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
st_audiorec = components.declare_component("st_audiorec", path=build_dir)

st.set_option('deprecation.showfileUploaderEncoding', False)

Languages = {'afrikaans':'af','albanian':'sq','amharic':'am','arabic':'ar','armenian':'hy','azerbaijani':'az','basque':'eu','belarusian':'be','bengali':'bn','bosnian':'bs','bulgarian':'bg','catalan':'ca','cebuano':'ceb','chichewa':'ny','chinese (simplified)':'zh-cn','chinese (traditional)':'zh-tw','corsican':'co','croatian':'hr','czech':'cs','danish':'da','dutch':'nl','english':'en','esperanto':'eo','estonian':'et','filipino':'tl','finnish':'fi','french':'fr','frisian':'fy','galician':'gl','georgian':'ka','german':'de','greek':'el','gujarati':'gu','haitian creole':'ht','hausa':'ha','hawaiian':'haw','hebrew':'iw','hebrew':'he','hindi':'hi','hmong':'hmn','hungarian':'hu','icelandic':'is','igbo':'ig','indonesian':'id','irish':'ga','italian':'it','japanese':'ja','javanese':'jw','kannada':'kn','kazakh':'kk','khmer':'km','korean':'ko','kurdish (kurmanji)':'ku','kyrgyz':'ky','lao':'lo','latin':'la','latvian':'lv','lithuanian':'lt','luxembourgish':'lb','macedonian':'mk','malagasy':'mg','malay':'ms','malayalam':'ml','maltese':'mt','maori':'mi','marathi':'mr','mongolian':'mn','myanmar (burmese)':'my','nepali':'ne','norwegian':'no','odia':'or','pashto':'ps','persian':'fa','polish':'pl','portuguese':'pt','punjabi':'pa','romanian':'ro','russian':'ru','samoan':'sm','scots gaelic':'gd','serbian':'sr','sesotho':'st','shona':'sn','sindhi':'sd','sinhala':'si','slovak':'sk','slovenian':'sl','somali':'so','spanish':'es','sundanese':'su','swahili':'sw','swedish':'sv','tajik':'tg','tamil':'ta','telugu':'te','thai':'th','turkish':'tr','turkmen':'tk','ukrainian':'uk','urdu':'ur','uyghur':'ug','uzbek':'uz','vietnamese':'vi','welsh':'cy','xhosa':'xh','yiddish':'yi','yoruba':'yo','zulu':'zu'}

translator = google_translator()

value1 = Languages['english']
value2 = Languages['arabic']

image = Image

#load pre-trained model
model = ResNet50(weights='imagenet')
#load Safae's model
#model = joblib.load('model.joblib')
#create file uploader

st.set_page_config(
    page_title = 'Matjari Vision',
    page_icon = ':innocent:'
)
st.header('Matjari Vision :innocent:')
#file uploader widget
img_data = st.file_uploader(label='Choose a file', type=['png', 'jpg', 'jpeg', 'heic'])

if img_data is not None:
    #display image
    uploaded_png = Image.open(img_data)
    st.image(uploaded_png)
    converted_png_img = uploaded_png.convert('RGB')
    img = asarray(converted_png_img)
    x = cv2.resize(img,(224,224),3)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prediction = model.predict(x)
    make_prediction =   decode_predictions(prediction)

    text1 = make_prediction[0][0][1]
    text2 = make_prediction[0][1][1]
    text3 = make_prediction[0][2][1]
    text4 = make_prediction[0][3][1]
    text5 = make_prediction[0][4][1]
    text8 = "Something Else: Record your answer below"

    #language = 'en'

    #speech1 = gTTS(text = text1, lang = language, slow = False)
    #speech2 = gTTS(text = text2, lang = language, slow = False)
    #speech3 = gTTS(text = text3, lang = language, slow = False)
    #speech4 = gTTS(text = text4, lang = language, slow = False)
    #speech5 = gTTS(text = text5, lang = language, slow = False)
    #speech8 = gTTS(text = text8, lang = language, slow = False)


    #speech1.save("text1.mp3")
    #speech2.save("text2.mp3")
    #speech3.save("text3.mp3")
    #speech4.save("text4.mp3")
    #speech5.save("text5.mp3")
    #speech8.save("text8.mp3")


    #audio_file11 = open('text1.mp3', 'rb')
    #audio_bytes11 = audio_file11.read()
    #audio_file22 = open('text2.mp3', 'rb')
    #audio_bytes22 = audio_file22.read()
    #audio_file33 = open('text3.mp3', 'rb')
    #audio_bytes33 = audio_file33.read()
    #audio_file44 = open('text4.mp3', 'rb')
    #audio_bytes44 = audio_file44.read()
    #audio_file55 = open('text4.mp3', 'rb')
    #audio_bytes55 = audio_file55.read()
    #audio_file88 = open('text8.mp3', 'rb')
    #audio_bytes88 = audio_file88.read()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.checkbox(make_prediction[0][0][1])
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.checkbox(make_prediction[0][1][1])
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.checkbox(make_prediction[0][2][1])
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.checkbox(make_prediction[0][3][1])
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.checkbox(make_prediction[0][4][1])
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.checkbox("Something Else: Record your answer below")


    #with col2:
        #st.markdown("***")
        #st.audio(audio_bytes11, format='audio/ogg')
        #st.text("")
        #st.audio(audio_bytes22, format='audio/ogg')
        #st.text("")
        #st.audio(audio_bytes33, format='audio/ogg')
        #st.text("")
        #st.audio(audio_bytes44, format='audio/ogg')
        #st.text("")
        #st.audio(audio_bytes55, format='audio/ogg')
        #st.text("")
        #st.audio(audio_bytes88, format='audio/ogg')

    with col2:
        result1 = translator.translate(text1, lang_tgt = 'ar')
        st.checkbox(result1)

        st.text("")
        st.text("")
        st.text("")
        st.text("")

        result2 = translator.translate(text2, lang_tgt = 'ar')
        st.checkbox(result2)

        st.text("")
        st.text("")
        st.text("")
        st.text("")
        result3 = translator.translate(text3, lang_tgt = 'ar')
        st.checkbox(result3)

        st.text("")
        st.text("")
        st.text("")
        st.text("")
        result4 = translator.translate(text4, lang_tgt = 'ar')
        st.checkbox(result4)

        st.text("")
        st.text("")
        st.text("")
        st.text("")
        result5 = translator.translate(text5, lang_tgt = 'ar')
        st.checkbox(result5)

        st.text("")
        st.text("")
        st.text("")
        st.text("")
        result8 = translator.translate(text8, lang_tgt = 'ar')
        st.checkbox(result8)

    with col3:
        speech1 = gTTS(result1, lang = 'ar', slow = False)
        speech1.save('user_trans1.mp3')
        audio_file1 = open('user_trans1.mp3', 'rb')
        audio_bytes1 = audio_file1.read()
        st.audio(audio_bytes1, format='audio/ogg',start_time=0)
        st.text("")

        speech2 = gTTS(result2, lang = 'ar', slow = False)
        speech2.save('user_trans2.mp3')
        audio_file2 = open('user_trans2.mp3', 'rb')
        audio_bytes2 = audio_file2.read()
        st.audio(audio_bytes2, format='audio/ogg',start_time=0)
        st.text("")

        speech3 = gTTS(result3, lang = 'ar', slow = False)
        speech3.save('user_trans3.mp3')
        audio_file3 = open('user_trans3.mp3', 'rb')
        audio_bytes3 = audio_file3.read()
        st.audio(audio_bytes3, format='audio/ogg',start_time=0)
        st.text("")
        st.text("")

        speech4 = gTTS(result4, lang = 'ar', slow = False)
        speech4.save('user_trans4.mp3')
        audio_file4 = open('user_trans4.mp3', 'rb')
        audio_bytes4 = audio_file4.read()
        st.audio(audio_bytes4, format='audio/ogg',start_time=0)
        st.text("")
        st.text("")

        speech5 = gTTS(result5, lang = 'ar', slow = False)
        speech5.save('user_trans5.mp3')
        audio_file5 = open('user_trans5.mp3', 'rb')
        audio_bytes5 = audio_file5.read()
        st.audio(audio_bytes5, format='audio/ogg',start_time=0)
        st.text("")
        st.text("")

        speech8 = gTTS(result8, lang = 'ar', slow = False)
        speech8.save('user_trans8.mp3')
        audio_file8 = open('user_trans8.mp3', 'rb')
        audio_bytes8 = audio_file8.read()
        st.audio(audio_bytes8, format='audio/ogg',start_time=0)


    st_audiorec()

        #translate1 = trans.translate(text1,lang_src=value1,lang_tgt=value2)
        #st.info(str(translate1))
        #converted_audio = gtts.gTTS(translate1, lang=value2)
        #converted_audio.save("translated1.mp3")
        #audio_file1 = open('translated1.mp3','rb')
        #audio_bytes1 = audio_file1.read()
        #st.audio(audio_bytes1, format='audio')

        #translate2 = translator.translate(text2,lang_src=value1,lang_tgt=value2)
        #st.info(str(translate2))
        #converted_audio2 = gtts.gTTS(translate2, lang=value2)
        #converted_audio2.save("translated2.mp3")
        #audio_file2 = open('translated2.mp3','rb')
        #audio_bytes2 = audio_file2.read()
        #st.audio(audio_bytes2, format='audio')

        #translate3 = translator.translate(text3,lang_src=value1,lang_tgt=value2)
        #st.info(str(translate3))
        #converted_audio3 = gtts.gTTS(translate3, lang=value2)
        #converted_audio3.save("translated3.mp3")
        #audio_file3 = open('translated3.mp3','rb')
        #audio_bytes3 = audio_file3.read()
        #st.audio(audio_bytes3, format='audio')

        #translate4 = translator.translate(text4,lang_src=value1,lang_tgt=value2)
        #st.info(str(translate4))
        #converted_audio4 = gtts.gTTS(translate4, lang=value2)
        #converted_audio4.save("translated4.mp3")
        #audio_file4 = open('translated4.mp3','rb')
        #audio_bytes4 = audio_file4.read()
        #st.audio(audio_bytes4, format='audio')

        #translate5 = translator.translate(text5,lang_src=value1,lang_tgt=value2)
        #st.info(str(translate5))
        #converted_audio5 = gtts.gTTS(translate5, lang=value2)
        #converted_audio5.save("translated5.mp3")
        #audio_file5 = open('translated5.mp3','rb')
        #audio_bytes5 = audio_file5.read()
        #st.audio(audio_bytes5, format='audio')

        #translate8 = translator.translate(text8,lang_src=value1,lang_tgt=value2)
        #st.info(str(translate8))
        #converted_audio8 = gtts.gTTS(translate8, lang=value2)
        #converted_audio8.save("translated8.mp3")
        #audio_file8 = open('translated8.mp3','rb')
        #audio_bytes8 = audio_file8.read()
        #st.audio(audio_bytes8, format='audio')

    #text = str in dir(make_prediction)
    #language = 'en'
    # speech = gTTS(text = text, lang = language, slow = False)
    # speech.save("text.mp3")
    # os.system("start text.mp3")
