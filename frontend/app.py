import streamlit as st
import requests
from PIL import Image
import io

# --- Streamlit UI ---
st.set_page_config(
    page_title="Medical AI Diagnosis System",
    page_icon="⚕️",
    layout="centered"
)

st.title("⚕️ AI-Powered Medical Image & Report Analysis")
st.markdown("""
Upload an X-ray/MRI image and provide a medical report or symptoms. 
Our AI system (powered by Google Gemini) will analyze the inputs to suggest a potential diagnosis.
""")

st.warning("""
**IMPORTANT MEDICAL DISCLAIMER:**
This AI system is for **informational purposes only** and is **NOT** a substitute for professional medical advice, diagnosis, or treatment. 
Always consult a qualified healthcare provider for any medical concerns. 
Do not disregard professional medical advice or delay in seeking it because of something you have used from this AI system.
""")

st.header("1. Upload X-ray/MRI Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

st.header("2. Enter Medical Report / Symptoms")
medical_report_text = st.text_area(
    "Briefly describe the patient's symptoms, relevant medical history, or details from a medical his/her report:",
    height=200,
    placeholder="e.g., 'Ex- Patient presents with persistent cough and shortness of breath for 3 days. Fever of 101°F. No known allergies. Non-smoker. X-ray taken today.'"
)

# --- Prediction Button ---
if st.button("Get AI Diagnosis"):
    if uploaded_file is None:
        st.error("Please upload an X-ray/MRI image.")
    elif not medical_report_text.strip():
        st.error("Please provide a medical report or symptoms.")
    else:
        with st.spinner("Analyzing...|.. This may take a moment."):
            try:
                # Prepare image for sending to FastAPI
                files = {'image': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Prepare text for sending to FastAPI
                data = {'medical_report': medical_report_text}
                
                # Make a POST request to the FastAPI backend
                # Ensure the URL matches your backend's host and port
                backend_url = "http://localhost:8000/predict" 
                response = requests.post(backend_url, files=files, data=data, timeout=120) # Added timeout for longer responses

                if response.status_code == 200:
                    result = response.json()
                    st.success("Analysis Complete!")
                    st.subheader("AI Suggested Diagnosis & Reasoning:")
                    st.markdown(f"```\n{result['diagnosis']}\n```")
                    st.info(result['disclaimer'])
                    
                    st.subheader("Uploaded Image:")
                    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

                else:
                    st.error(f"Error from backend: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend API. Please ensure the backend is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.markdown("Developed with ❤️ using FastAPI, Streamlit, and Google Gemini AI.")
