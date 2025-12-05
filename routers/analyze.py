from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from services.grain_analyzer import analyze_image

router = APIRouter(prefix="/api", tags=["Analysis"])

@router.post("/analyze/physical", summary="Analyze rice grain image for physical grading")
async def analyze_grain_image(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / image.filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        result = analyze_image(str(temp_path))

        result["timestamp"] = datetime.utcnow().isoformat()
        result["modelVersion"] = "grain-physical-v1"
        result["filename"] = image.filename

        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)