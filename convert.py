

import streamlit as st

audio_file = open('sample-audio-files/Abnormal-HeartBeat1.wav','rb') #enter the filename with filepath

audio_bytes = audio_file.read() #reading the file

st.audio(audio_bytes, format='audio/ogg')