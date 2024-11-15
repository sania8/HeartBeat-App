import streamlit as st

# Play the audio using Streamlit without loop
st.audio("f0015.mp3", format="audio/mpeg")

# Add HTML to loop the audio
st.markdown(
    """
    <audio autoplay loop>
        <source src="f0015.mp3" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    """, unsafe_allow_html=True
)
