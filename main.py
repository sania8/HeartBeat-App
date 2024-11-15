import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Assuming the actual labels are stored in a dictionary
true_labels = {
    "sample1.wav": "Normal",
    "sample2.wav": "Abnormal",
    "sample3.wav": "Normal",
    "sample4.wav": "Abnormal",
    "sample5.wav": "Normal",
    "sample6.wav": "Abnormal",
}

st.set_page_config(
    page_title="My YOLO App",
    page_icon="🚀"
)

# Remove whitespace from the top of the page and sidebar
st.markdown("""
    <style>
        body {
            padding-left: 3rem;
            padding-right: 3rem;
        }
        .reportview-container {
            background-color: pink;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem; 
        }
        .st-emotion-cache-ocqkz7 {
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
        }
        .stApp {
            margin-left: 0;
        }
        .column {
            flex: 1;
            margin: 0.5rem;
        }
        .column img {
            width: 100%;
            height: auto;
        }
        .title {
            padding-top: 4rem; /* Add margin to the top of the title */
        }
    </style>
""", unsafe_allow_html=True)

st.title('**Heartbeat Sound Classifier**')

# Load the YOLO model
model = YOLO('best.pt')  # Use the correct path to your pre-trained model

# Create three columns for layout
col_left, col_intro, col_right = st.columns([2, 3, 4])  # Adjust column widths as needed

# Direct list of audio file paths for sample audio files (WAV and MP3)
audio_files = {
    "Normal-HeartBeat1": {
        "wav": "sample-audio-files/Normal-HeartBeat1.wav",
        "mp3": "sample-audio-files/Normal-HeartBeat1.mp3",
    },
    "Abnormal-HeartBeat1": {
        "wav": "sample-audio-files/Abnormal-HeartBeat1.wav",
        "mp3": "sample-audio-files/Abnormal-HeartBeat1.mp3",
    },
    "Normal-HeartBeat2": {
        "wav": "sample-audio-files/Normal-HeartBeat2.wav",
        "mp3": "sample-audio-files/Normal-HeartBeat2.mp3",
    },
    "Abnormal-HeartBeat2": {
        "wav": "sample-audio-files/Abnormal-HeartBeat2.wav",
        "mp3": "sample-audio-files/Abnormal-HeartBeat2.mp3",
    },
    "Normal-HeartBeat3": {
        "wav": "sample-audio-files/Normal-HeartBeat3.wav",
        "mp3": "sample-audio-files/Normal-HeartBeat3.mp3",
    },
    "Abnormal-HeartBeat3": {
        "wav": "sample-audio-files/Abnormal-HeartBeat3.wav",
        "mp3": "sample-audio-files/Abnormal-HeartBeat3.mp3",
    },
}

with col_left:
    st.markdown("### Sample Data for Testing")
    selected_audio = st.selectbox(
        "Choose a sample audio file:",
        [""] + list(audio_files.keys())  # Dropdown with file names
    )

    # Map selected name to actual file paths (WAV and MP3) if one is selected
    selected_audio_paths = None
    if selected_audio:
        selected_audio_paths = audio_files[selected_audio]
        st.write(f"**Attached File:** {selected_audio}.wav")  # Display the WAV file name
        st.audio(selected_audio_paths["mp3"])  # Play the corresponding MP3 file

with col_intro:
    st.markdown(
        """
        <div style="margin-top: 0rem;">
            <h3>Introduction</h3>
            <p style="margin-top:-1rem;">Welcome to the Heartbeat Sound Classification App. This application is designed to classify heartbeat sounds as Normal/Abnormal with an accuracy of 94.6%.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style="margin-top: -1rem;">
            <h4>How to Use:</h4>
            <p style="margin-top:-1rem;">1. Upload a WAV audio file using the file uploader below.</p>
            <p style="margin-top:-1rem;">2. Click the 'Predict' button to analyze the uploaded audio.</p>
            <p style="margin-top:-1rem;">3. The predictions include the classification (Normal/Abnormal) and the confidence level.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col_right:
    # File uploader for custom uploaded audio
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

    # Check if a custom file is uploaded or if a sample is selected
    if uploaded_file:
        # If uploaded file is present, use it
        file_name = os.path.basename(uploaded_file.name)
        st.write(f"**Attached File:** {file_name}")
        audio_file = uploaded_file
    elif selected_audio_paths:
        # If a sample audio is selected from the dropdown, use it
        file_name = os.path.basename(selected_audio_paths["wav"])
        st.write(f"**Attached File:** {file_name}")
        audio_file = selected_audio_paths["wav"]
    else:
        # If neither a file nor a sample is selected, display nothing
        st.write("**Attached File:** None")
        audio_file = None

    # Predict button
    if st.button("Predict"):
        if audio_file is not None:
            if isinstance(audio_file, bytes):  # If the file is uploaded via the file uploader
                audio_data = wavfile.read(audio_file)
            else:
                rate, data = wavfile.read(audio_file)

            # Generate and save the spectrogram
            plt.specgram(data, Fs=rate, aspect='auto')
            plt.xlim(0, 7)
            plt.ylim(0, 1000)

            # Remove ticks
            plt.xticks([])
            plt.yticks([])

            plt.savefig('temp.png')
            spectrogram = Image.open('temp.png')

            # Make predictions
            predictions = model.predict(spectrogram)[0].probs

            if predictions.data[0] > predictions.data[1]:
                class__ = 'Abnormal'
                conf__ = predictions.data[0]
            else:
                class__ = "Normal"
                conf__ = predictions.data[1]

            st.write("Predicted class:", class__, "(confidence:", round(float(conf__) * 100, 2), "%)")


            # Create content for the prediction file
            content = f"Predicted class: {class__}\nConfidence: {round(float(conf__) * 100, 2)}%\n"

            # Save predictions to a text file
            with open("predictions.txt", "w") as f:
                f.write(content)

            # Download predictions file
            st.download_button(
                label="Download Predictions",
                data=content,
                file_name="heartbeat_predictions.txt",
                mime="text/plain"
            )
        else:
            st.warning("Please upload a file or select a sample first.")
