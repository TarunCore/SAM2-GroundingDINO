import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import base64
import whisper
import sounddevice as sd
import tempfile
import os
from streamlit_drawable_canvas import st_canvas
import json
import requests
from io import BytesIO

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'click_coordinates' not in st.session_state:
    st.session_state.click_coordinates = []
if 'voice_text' not in st.session_state:
    st.session_state.voice_text = ""
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = None
if 'result_mask' not in st.session_state:
    st.session_state.result_mask = None

# API endpoint configuration
API_ENDPOINT = "http://localhost:5000/process"

@st.cache_resource
def load_whisper_model():
    """Load Whisper model for voice-to-text conversion"""
    return whisper.load_model("base")

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone"""
    st.info(f"Recording for {duration} seconds... Speak now!")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()  # Wait until recording is finished
    return audio_data.flatten(), sample_rate

def transcribe_audio(audio_data, sample_rate):
    """Transcribe audio using Whisper"""
    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        # Convert to int16 for wav format
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Write wav file manually or use scipy if available
        try:
            from scipy.io.wavfile import write
            write(tmp_file.name, sample_rate, audio_int16)
        except ImportError:
            # Fallback: use opencv to write audio (less ideal)
            st.error("scipy not available. Please install scipy for audio processing.")
            return ""
    
    try:
        # Load Whisper model and transcribe
        model = load_whisper_model()
        result = model.transcribe(tmp_file.name)
        transcribed_text = result['text'].strip()
        
        # Clean up temporary file
        os.unlink(tmp_file.name)
        
        return transcribed_text
    except Exception as e:
        st.error(f"Error in transcription: {str(e)}")
        return ""

def process_image_click(image, click_point):
    """Process image with SAM using click coordinates"""
    # Placeholder for SAM integration
    st.success(f"Touch/Click detected at coordinates: {click_point}")
    st.info("This will be connected to SAM for mask generation")
    
    # You'll replace this with your SAM integration
    # Example:
    # masks = your_sam_model.predict(image, click_point)
    # return masks
    
    return f"SAM processing for point {click_point}"

def process_image_with_text(image, text_prompt):
    """Process image with GroundingDINO using text prompt"""
    try:
        # Convert PIL Image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Prepare the files and data for the request
        files = {
            'image': ('image.png', img_byte_arr, 'image/png')
        }
        data = {
            'prompt': text_prompt
        }

        # Make the API request
        response = requests.post(API_ENDPOINT, files=files, data=data)
        
        if response.status_code == 200:
            # Convert the response content (mask image) to numpy array
            mask_bytes = BytesIO(response.content)
            mask_image = Image.open(mask_bytes)
            st.session_state.result_mask = mask_image
            return True
        else:
            st.error(f"API Error: {response.text}")
            return False

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return False

def main():
    st.set_page_config(page_title="Multi-Modal Computer Vision UI", layout="wide")
    
    st.title("🎯 Multi-Modal Computer Vision Interface")
    st.markdown("Upload an image and interact using **Touch/Click**, **Voice**, or **Text** input")
    
    # Sidebar for model status and settings
    with st.sidebar:
        st.header("Model Status")
        st.success("✅ Whisper (Voice-to-Text)")
        st.info("🔄 SAM (Ready for integration)")
        st.info("🔄 GroundingDINO (Ready for integration)")
        
        st.header("Settings")
        recording_duration = st.slider("Voice recording duration (seconds)", 3, 10, 5)
    
    # Image upload section
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        
        # Create two columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Image & Interaction")
            
            # Display image with clickable canvas
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=3,
                stroke_color="#FF0000",
                background_image=image,
                update_streamlit=True,
                height=min(500, image.height),
                width=min(700, image.width),
                drawing_mode="point",
                point_display_radius=8,
                key="canvas",
            )
            
            # Process canvas clicks
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                if objects and len(objects) > 0:
                    # Get the last clicked point
                    last_point = objects[-1]
                    if last_point["type"] == "circle":
                        click_x = int(last_point["left"])
                        click_y = int(last_point["top"])
                        st.session_state.click_coordinates = [click_x, click_y]
                        st.session_state.processing_mode = "touch"
        
        with col2:
            st.subheader("Input Methods")
            
            # Method 1: Touch/Click (already handled above)
            st.markdown("**1. 👆 Touch/Click Mode**")
            if st.session_state.click_coordinates:
                st.success(f"Point selected: {st.session_state.click_coordinates}")
                if st.button("🎯 Process with SAM", key="sam_process"):
                    result = process_image_click(image, st.session_state.click_coordinates)
                    st.write(result)
            else:
                st.info("Click on the image to select a point")
            
            st.divider()
            
            # Method 2: Voice input
            st.markdown("**2. 🎤 Voice Input**")
            col_voice1, col_voice2 = st.columns(2)
            
            with col_voice1:
                if st.button("🎙️ Start Recording"):
                    try:
                        audio_data, sample_rate = record_audio(recording_duration)
                        transcribed_text = transcribe_audio(audio_data, sample_rate)
                        if transcribed_text:
                            st.session_state.voice_text = transcribed_text
                            st.session_state.processing_mode = "voice"
                            st.success("Recording complete!")
                        else:
                            st.error("No speech detected or transcription failed")
                    except Exception as e:
                        st.error(f"Recording error: {str(e)}")
            
            with col_voice2:
                if st.session_state.voice_text:
                    st.text_area("Transcribed:", st.session_state.voice_text, height=60, disabled=True)
                    if st.button("🔍 Process Voice Text", key="voice_process"):
                        result = process_image_with_text(image, st.session_state.voice_text)
                        st.write(result)
            
            st.divider()
            
            # Method 3: Text input
            st.markdown("**3. ✏️ Text Input**")
            text_prompt = st.text_input("Enter your prompt:", placeholder="e.g., 'person', 'car', 'dog'")
            
            if st.button("🔍 Process Text Prompt", key="text_process"):
                if text_prompt.strip():
                    st.session_state.processing_mode = "text"
                    result = process_image_with_text(image, text_prompt)
                    st.write(result)
                else:
                    st.warning("Please enter a text prompt")
        
        # Results section
        st.divider()
        st.subheader("Processing Results")
        
        if st.session_state.processing_mode:
            st.info(f"Last processing mode: **{st.session_state.processing_mode.upper()}**")
            
            # Display results based on processing mode
            if st.session_state.processing_mode == "touch":
                st.markdown("**SAM Results:** Mask generation from click point")
            elif st.session_state.processing_mode in ["voice", "text"]:
                st.markdown("**GroundingDINO + SAM Results:** Object detection and segmentation")
                if st.session_state.result_mask is not None:
                    # Display the original image and mask side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Original Image")
                    with col2:
                        st.image(st.session_state.result_mask, caption="Generated Mask")
    
    else:
        st.info("👆 Please upload an image to get started")
        
        # Show example of what the interface will do
        st.markdown("""
        ### How it works:
        1. **Touch/Click Mode**: Click anywhere on the image → Directly to SAM for mask generation
        2. **Voice Mode**: Record your voice → Whisper converts to text → GroundingDINO processes
        3. **Text Mode**: Type your prompt → GroundingDINO processes → SAM generates masks
        """)

if __name__ == "__main__":
    main()