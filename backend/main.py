from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the Gemini 1.5 Flash model
try:
    model = genai.GenerativeModel('gemini-2.0-flash') # Using gemini-1.5-flash as requested
    print("Gemini 1.5 Flash model initialized successfully.")
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    print("Please ensure your API key has access to '	gemini-2.5-flash-preview-05-20' and the model is available in your region.")
    # In a production app, you might want more robust error handling or exit here.

app = FastAPI(
    title="Medical Image AI Predictor Backend",
    description="Analyzes X-ray/MRI images and medical reports using Google Gemini AI."
)

@app.post("/predict")
async def predict_disease(
    image: UploadFile = File(..., description="X-ray or MRI image to analyze"),
    medical_report: str = Form(..., description="Medical report or symptoms in text format")
):
    """
    Analyzes an X-ray/MRI image and a medical report to predict diseases using Gemini AI.
    """
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="Gemini API Key not configured on the backend.")

    try:
        # Read image data
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))

        # Prepare content for Gemini 1.5 Flash
        # The prompt remains similar, as 1.5-flash is also multimodal.
        prompt = [
            "Analyze the provided medical image (X-ray/MRI) and the accompanying medical report/symptoms. "
            "Based on this information, provide a likely disease diagnosis. "
            "Explain your reasoning concisely, highlighting key findings from both the image and the text. "
            "Also, mention any important differential diagnoses if applicable.\n\n"
            "Medical Report/Symptoms: ",
            medical_report,
            "\n\nImage:",
            pil_image
        ]

        # Generate content using Gemini
        print("Sending request to Gemini 1.5 Flash API...")
        response = model.generate_content(prompt)
        print("Received response from Gemini 1.5 Flash API.")

        # Extract text from the response
        diagnosis_text = response.text

        return JSONResponse(content={
            "diagnosis": diagnosis_text,
            "disclaimer": "This AI system is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider."
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process request: {e}")
    
if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8000)

    