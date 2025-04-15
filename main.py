from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import pdfplumber
import spacy
import tensorflow as tf
import io
import os

# Fix for Keras 3 compatibility issue in Transformers
os.environ["TF_KERAS"] = "1"

# Ensure tf-keras is used instead of Keras 3
try:
    import tf_keras as keras
except ModuleNotFoundError:
    raise Exception("tf-keras not found. Install it using: pip install tf-keras")

# Load SciSpaCy Medical Model
try:
    nlp = spacy.load("en_core_sci_lg")
except:
    raise Exception("SciSpaCy model not found. Install it using the correct command.")

# Load Text Summarization Model
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize FastAPI App
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Medical Report Reader API is running!"}

def extract_text_from_image(file):
    image = Image.open(io.BytesIO(file.read()))
    text = pytesseract.image_to_string(image)
    return text.strip()

def extract_text_from_pdf(file):
    text = ""
    pdf_bytes = io.BytesIO(file.read())

    with pdfplumber.open(pdf_bytes) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    if not text.strip():
        text = ocr_pdf(pdf_bytes)

    return text.strip()

def ocr_pdf(pdf_bytes):
    images = pdfplumber.open(pdf_bytes).images
    ocr_text = ""
    
    for img in images:
        ocr_text += pytesseract.image_to_string(Image.open(io.BytesIO(img))).strip() + "\n"
    
    return ocr_text.strip()

def process_medical_text(text):
    if not text:
        return {"error": "No text detected. Ensure the file is clear."}

    # Get summary first
    max_input_length = 1024
    text_to_summarize = text[:max_input_length]
    summary = summarizer(text_to_summarize, max_length=150, min_length=50, do_sample=False)
    summary_text = summary[0]['summary_text']

    # Then process entities
    doc = nlp(text)
    medical_terms = {ent.text: ent.label_ for ent in doc.ents}

    return {
        #"summary": summary_text,  # Summary comes first
        "summary": summary[0]['summary_text'],
        "extracted_text": text,
        "medical_entities": medical_terms
    }

@app.post("/upload/")
async def upload_report(file: UploadFile = File(...)):
    try:
        filename = file.filename.lower()

        if filename.endswith((".png", ".jpg", ".jpeg")):
            extracted_text = extract_text_from_image(file.file)
        elif filename.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(file.file)
        else:
            return {"error": "Unsupported file format. Upload JPG, PNG, or PDF."}

        return process_medical_text(extracted_text)

    except Exception as e:
        return {"error": str(e)}
    
# To initiate the Server/ FastApi 
# Type Command in terminal: uvicorn main:app --reload 
# Then Launch the website