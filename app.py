from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import json
from main import IntegratedMedicalOCR
import zipfile

# Unzip the medical JSON if not already extracted
json_path = "data/medical_products_full.json"
zip_path = "data/medical_products_full.json.zip"

if not os.path.exists(json_path) and os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/")

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Medical OCR API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR system
ocr_system = None

@app.on_event("startup")
async def startup_event():
    global ocr_system
    
    config = {
        "llama_model_name": os.getenv("LLAMA_MODEL_NAME", "elenamagdy77/Finetune_Llama_3_2_Vision_OCR"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "medicine_csv_path": "data/eda_medicines_cleaned.csv",
        "dosage_json_path": "data/medical_products_full.json"
    }
    
    try:
        ocr_system = IntegratedMedicalOCR(**config)
        print("✅ OCR System initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize OCR system: {e}")

@app.get("/")
async def root():
    return {"message": "Medical OCR API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "ocr_system_ready": ocr_system is not None}

@app.post("/process-prescription")
async def process_prescription(file: UploadFile = File(...)):
    if not ocr_system:
        raise HTTPException(status_code=503, detail="OCR system not initialized")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process the prescription
        results = ocr_system.process_prescription(tmp_file_path)
        formatted_results = ocr_system.format_results(results)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return {
            "success": True,
            "filename": file.filename,
            "results": formatted_results
        }
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))