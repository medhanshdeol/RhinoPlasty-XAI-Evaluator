import os
import cv2  # type: ignore
from fastapi import FastAPI, UploadFile, File, Form, HTTPException  # type: ignore
from fastapi.responses import JSONResponse, FileResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
import tempfile
import uuid

from preprocess.mediapipe_extractor import FaceLandmarkExtractor
from analysis.differential_analyzer import DifferentialAnalyzer
from xai.anomaly_visualizer import generate_discrepancy_heatmap

app = FastAPI(title="Differential Anomaly Detection API")

# Mount the static directory for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

extractor = FaceLandmarkExtractor()
analyzer = DifferentialAnalyzer()

# Make sure temp directory exists
os.makedirs("temp", exist_ok=True)

@app.post("/analyze_differential")
async def analyze_differential(
    planned_img: UploadFile = File(...),
    actual_img: UploadFile = File(...),
    patient_satisfaction: float = Form(5.0),
    plan_depth: UploadFile = File(None),
    actual_depth: UploadFile = File(None)
):
    try:
        # Save uploaded files temporarily
        plan_path = f"temp/{uuid.uuid4()}_{planned_img.filename}"
        actual_path = f"temp/{uuid.uuid4()}_{actual_img.filename}"
        
        with open(plan_path, "wb") as buffer:
            buffer.write(await planned_img.read())
        with open(actual_path, "wb") as buffer:
            buffer.write(await actual_img.read())
            
        plan_depth_path = None
        if plan_depth:
            plan_depth_path = f"temp/{uuid.uuid4()}_{plan_depth.filename}"
            with open(plan_depth_path, "wb") as buffer:
                buffer.write(await plan_depth.read())
                
        actual_depth_path = None
        if actual_depth:
            actual_depth_path = f"temp/{uuid.uuid4()}_{actual_depth.filename}"
            with open(actual_depth_path, "wb") as buffer:
                buffer.write(await actual_depth.read())
            
        # 1. Extract Landmarks
        plan_res = extractor.extract_landmarks(plan_path, plan_depth_path)
        actual_res = extractor.extract_landmarks(actual_path, actual_depth_path)
        
        if plan_res is None or actual_res is None:
            raise HTTPException(status_code=400, detail="Could not detect faces in one or both images.")
            
        plan_landmarks_mm, _ = plan_res
        actual_landmarks_mm, actual_landmarks_norm = actual_res
            
        # 2. Align Actual to Plan
        # We align the actual landmarks to match the planned landmarks' scale and position
        actual_aligned = extractor.procrustes_align(actual_landmarks_mm, plan_landmarks_mm)
        
        # 3. Compute Discrepancies
        discrepancies, vectors = analyzer.compute_discrepancies(plan_landmarks_mm, actual_aligned)
        
        # 4. Regional Scores
        regional_scores = analyzer.compute_regional_discrepancies(discrepancies)
        
        # 5. Generate Heatmap
        heatmap_path = f"temp/heatmap_{uuid.uuid4()}.jpg"
        # We use the original actual_landmarks for visualization so it overlays correctly on the actual image
        generate_discrepancy_heatmap(actual_path, actual_landmarks_norm, discrepancies, vectors, heatmap_path)
        
        # Cleanup input files
        if os.path.exists(plan_path):
            os.remove(plan_path)
        if os.path.exists(actual_path):
            os.remove(actual_path)
        if plan_depth_path and os.path.exists(plan_depth_path):
            os.remove(plan_depth_path)
        if actual_depth_path and os.path.exists(actual_depth_path):
            os.remove(actual_depth_path)
        
        # Compute a combined holistic score (70% objective SFI, 30% subjective satisfaction)
        holistic_score = (regional_scores.get('SFI_overall', 0) * 0.7) + (patient_satisfaction * 10 * 0.3)
        
        return {
            "status": "success",
            "regional_scores": regional_scores,
            "patient_satisfaction": patient_satisfaction,
            "holistic_score": holistic_score,
            "heatmap_path": heatmap_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_heatmap")
async def download_heatmap(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
