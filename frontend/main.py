import streamlit as st
import numpy as np
from PIL import Image
import whisper
import sounddevice as sd
import tempfile
import os
import cv2
import requests
import io

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = None
if 'click_point' not in st.session_state:
    st.session_state.click_point = None
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""
if 'voice_text' not in st.session_state:
    st.session_state.voice_text = ""
if 'result_image' not in st.session_state:
    st.session_state.result_image = None

# Backend API endpoint
BACKEND_URL = "http://52.204.25.120:5000/process"

@st.cache_resource
def load_whisper_model():
    """Load Whisper model"""
    return whisper.load_model("base")

def process_image_with_prompt(image, prompt):
    """Send image and prompt to backend API"""
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Prepare the files and data for the request
    files = {
        'image': ('image.png', img_byte_arr, 'image/png')
    }
    data = {
        'prompt': prompt
    }

    try:
        # Make POST request to backend
        response = requests.post(BACKEND_URL, files=files, data=data)
        
        if response.status_code == 200:
            # Convert response content to image
            mask_image = Image.open(io.BytesIO(response.content))
            return mask_image
        else:
            st.error(f"Error from backend: {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to backend: {str(e)}")
        return None

def record_and_transcribe():
    """Record audio and convert to text"""
    st.info("Recording for 5 seconds... Speak now!")
    
    # Record audio
    duration = 5
    sample_rate = 16000
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    
    # Save to temp file and transcribe
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        audio_int16 = (audio_data.flatten() * 32767).astype(np.int16)
        
        try:
            from scipy.io.wavfile import write
            write(tmp_file.name, sample_rate, audio_int16)
        except ImportError:
            st.error("Please install scipy: pip install scipy")
            return ""
    
    try:
        model = load_whisper_model()
        result = model.transcribe(tmp_file.name)
        os.unlink(tmp_file.name)
        return result['text'].strip()
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return ""

def main():
    st.title("🎯 Multi-Modal Computer Vision Interface")
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        
        # Create two columns for original and result images
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        
        with col2:
            if st.session_state.result_image is not None:
                st.image(st.session_state.result_image, caption="Segmentation Mask", use_column_width=True)
        
        # Three input options
        st.markdown("### Choose Input Method:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("👆 Touch", use_container_width=True):
                st.session_state.input_mode = "touch"
                st.session_state.result_image = None
        
        with col2:
            if st.button("🎤 Voice", use_container_width=True):
                st.session_state.input_mode = "voice"
                st.session_state.result_image = None
        
        with col3:
            if st.button("✏️ Text", use_container_width=True):
                st.session_state.input_mode = "text"
                st.session_state.result_image = None
        
        # Show input interface based on selection
        if st.session_state.input_mode == "touch":
            st.markdown("### 👆 Touch Mode - Select coordinates")
            
            # Get image dimensions
            img_width, img_height = image.size
            
            # Manual coordinate input
            st.write(f"Image size: {img_width} x {img_height}")
            
            col_x, col_y = st.columns(2)
            with col_x:
                click_x = st.number_input("X coordinate", min_value=0, max_value=img_width-1, value=img_width//2)
            with col_y:
                click_y = st.number_input("Y coordinate", min_value=0, max_value=img_height-1, value=img_height//2)
            
            if st.button("Preview Point"):
                st.session_state.click_point = [int(click_x), int(click_y)]
                img_array = np.array(image)
                img_with_dot = cv2.circle(img_array.copy(), (int(click_x), int(click_y)), 10, (255, 0, 0), -1)
                st.image(img_with_dot, caption=f"Selected point: ({int(click_x)}, {int(click_y)})", use_column_width=True)
        
        elif st.session_state.input_mode == "voice":
            st.markdown("### 🎤 Voice Mode")
            
            if st.button("🎙️ Start Recording"):
                with st.spinner("Recording..."):
                    transcribed = record_and_transcribe()
                    if transcribed:
                        st.session_state.voice_text = transcribed
                        st.success("Recording complete!")
            
            if st.session_state.voice_text:
                st.text_area("Transcribed Text:", st.session_state.voice_text, height=100)
                
                if st.button("Process with Voice Prompt"):
                    with st.spinner("Processing image..."):
                        result = process_image_with_prompt(image, st.session_state.voice_text)
                        if result is not None:
                            st.session_state.result_image = result
                            st.experimental_rerun()
        
        elif st.session_state.input_mode == "text":
            st.markdown("### ✏️ Text Mode")
            
            text_prompt = st.text_input("Enter your prompt:", placeholder="e.g., person, car, dog")
            
            if text_prompt and st.button("Process with Text Prompt"):
                with st.spinner("Processing image..."):
                    result = process_image_with_prompt(image, text_prompt)
                    if result is not None:
                        st.session_state.result_image = result
                        st.experimental_rerun()
    
    else:
        st.info("Please upload an image to start")

if __name__ == "__main__":
    main()