#!/usr/bin/env python3
"""
FastAPI app for Medical OCR Pipeline deployment on Railway
Uses the FixedMedicalOCR class for better accuracy and fewer false positives
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
import json
import logging
from typing import Dict, Any
import asyncio
from contextlib import asynccontextmanager
import zipfile

# Unzip the medical JSON if not already extracted
json_path = "data/medical_products_full.json"
zip_path = "data/medical_products_full.json.zip"

if not os.path.exists(json_path) and os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/")


# Import the fixed OCR system
from integrate import FixedMedicalOCR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global OCR system instance
ocr_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global ocr_system
    
    # Startup
    logger.info("🚀 Starting Medical OCR API...")
    
    config = {
        "llama_model_name": os.getenv("LLAMA_MODEL_NAME", "elenamagdy77/Finetune_Llama_3_2_Vision_OCR"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "medicine_csv_path": "data/eda_medicines_cleaned.csv",
        "dosage_json_path": "data/medical_products_full.json"
    }
    
    # Validate required environment variables
    missing_vars = []
    if not config["google_api_key"]:
        missing_vars.append("GOOGLE_API_KEY")
    if not config["openai_api_key"]:
        missing_vars.append("OPENAI_API_KEY")
    
    if missing_vars:
        logger.warning(f"⚠️ Missing environment variables: {missing_vars}")
        logger.warning("Some OCR features may not work properly")
    
    # Check if data files exist
    data_files_missing = []
    if not os.path.exists(config["medicine_csv_path"]):
        data_files_missing.append("medicine database")
        logger.warning(f"⚠️ Medicine database not found at {config['medicine_csv_path']}")
    
    if not os.path.exists(config["dosage_json_path"]):
        data_files_missing.append("dosage database")
        logger.warning(f"⚠️ Dosage database not found at {config['dosage_json_path']}")
    
    try:
        logger.info("🔄 Initializing OCR system...")
        ocr_system = FixedMedicalOCR(**config)
        logger.info("✅ OCR System initialized successfully!")
        
        # Test the system with a simple check
        if hasattr(ocr_system, 'medicine_df') and ocr_system.medicine_df is not None:
            logger.info(f"📊 Medicine database loaded with {len(ocr_system.medicine_df)} entries")
        else:
            logger.warning("⚠️ Medicine database not properly loaded")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize OCR system: {e}")
        logger.error("API will run in limited mode")
        ocr_system = None
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Medical OCR API...")
    ocr_system = None

# Create FastAPI app with lifespan
app = FastAPI(
    title="Medical OCR API", 
    version="2.0.0",
    description="Advanced Medical Prescription OCR with LLaMA Vision and Google Vision API",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler for better error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Medical OCR API is running",
        "status": "healthy",
        "version": "2.0.0",
        "features": [
            "LLaMA Vision OCR",
            "Google Vision API",
            "Arabic text processing",
            "Medicine database matching",
            "Default dosage lookup"
        ],
        "endpoints": {
            "/health": "Health check",
            "/process-prescription": "Process prescription image",
            "/process-simple": "Process with simplified output",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    health_status = {
        "status": "healthy",
        "ocr_system_ready": ocr_system is not None,
        "components": {}
    }
    
    if ocr_system:
        # Check LLaMA model
        health_status["components"]["llama_vision"] = {
            "available": ocr_system.llama_model is not None,
            "status": "ready" if ocr_system.llama_model else "not_loaded"
        }
        
        # Check Google Vision
        health_status["components"]["google_vision"] = {
            "available": ocr_system.google_api_key is not None,
            "status": "ready" if ocr_system.google_api_key else "no_api_key"
        }
        
        # Check OpenAI
        health_status["components"]["openai"] = {
            "available": ocr_system.openai_client is not None,
            "status": "ready" if ocr_system.openai_client else "not_configured"
        }
        
        # Check databases
        health_status["components"]["medicine_database"] = {
            "available": ocr_system.medicine_df is not None,
            "count": len(ocr_system.medicine_df) if ocr_system.medicine_df is not None else 0
        }
        
        health_status["components"]["dosage_database"] = {
            "available": bool(ocr_system.dosage_db),
            "count": len(ocr_system.dosage_db) if ocr_system.dosage_db else 0
        }
    else:
        health_status["status"] = "unhealthy"
        health_status["error"] = "OCR system not initialized"
    
    return health_status

@app.post("/process-prescription")
async def process_prescription(file: UploadFile = File(...)):
    """Process prescription image and return detailed results"""
    if not ocr_system:
        raise HTTPException(
            status_code=503, 
            detail="OCR system not initialized. Check server logs for details."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"File must be an image. Received: {file.content_type}"
        )
    
    # Check file size (limit to 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    
    tmp_file_path = None
    try:
        # Read file content
        content = await file.read()
        
        if len(content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {max_size / (1024*1024)}MB"
            )
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"📁 Processing file: {file.filename} ({len(content)} bytes)")
        
        # Process the prescription
        results = ocr_system.process_prescription(tmp_file_path)
        formatted_results = ocr_system.format_results(results)
        
        logger.info(f"✅ Successfully processed {file.filename}: {len(results)} medicines found")
        
        return {
            "success": True,
            "filename": file.filename,
            "file_size_bytes": len(content),
            "processing_completed": True,
            "results": formatted_results,
            "metadata": {
                "total_medicines_found": len(results),
                "confidence_threshold_used": 0.7,
                "processing_method": "fixed_medical_ocr"
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Error processing {file.filename}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Processing failed: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
                logger.debug(f"🗑️ Cleaned up temporary file: {tmp_file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clean up temporary file: {e}")

@app.post("/process-simple")
async def process_prescription_simple(file: UploadFile = File(...)):
    """Process prescription and return simplified JSON format"""
    if not ocr_system:
        raise HTTPException(
            status_code=503, 
            detail="OCR system not initialized"
        )
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image"
        )
    
    tmp_file_path = None
    try:
        # Read and save file
        content = await file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process prescription
        results = ocr_system.process_prescription(tmp_file_path)
        
        # Create simplified format directly
        simplified_results = []
        for i, result in enumerate(results, 1):
            simplified_results.append({
                "rank": i,
                "medicine": result.medicine,
                "dosage": result.dosage,
                "default_dosage": result.default_dosage
            })
        
        return {
            "success": True,
            "filename": file.filename,
            "medicines": simplified_results
        }
        
    except Exception as e:
        logger.error(f"❌ Error in simple processing: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Processing failed: {str(e)}"
        )
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass

@app.get("/system-info")
async def get_system_info():
    """Get detailed system information"""
    info = {
        "api_version": "2.0.0",
        "python_version": os.sys.version,
        "environment": {
            "PORT": os.getenv("PORT", "8000"),
            "LLAMA_MODEL_NAME": os.getenv("LLAMA_MODEL_NAME", "Not set"),
            "HAS_GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
            "HAS_OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        },
        "data_files": {
            "medicine_csv": os.path.exists("data/eda_medicines_cleaned.csv"),
            "dosage_json": os.path.exists("data/medical_products_full.json")
        },
        "ocr_system_status": "initialized" if ocr_system else "not_initialized"
    }
    
    return info

# Add a simple test endpoint for development
@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {
        "message": "Test endpoint working",
        "system_ready": ocr_system is not None,
        "timestamp": str(asyncio.get_event_loop().time())
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Run the application
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True
    )