# Rhinoplasty XAI Evaluator

A novel, accessible, and explainable AI (XAI) evaluation pipeline for aesthetic surgery. This project bridges the gap between purely objective physical millimeter changes and subjective patient-reported outcomes (PROMs), providing a comprehensive "Holistic Score" for rhinoplasty success.

By leveraging **RGB-D (Color + True Depth)** capabilities, this tool bypasses the need for expensive, proprietary 3D clinical scanners (like VECTRA) by allowing surgeons to utilize standard smartphone LiDAR or TrueDepth camera captures to evaluate volumetric tissue displacement.

## Key Features

1. **RGB-D Volumetric Analysis:** 
   - Uses MediaPipe Face Mesh (CNN) to map 478 precise facial topological points on standard 2D color images.
   - Accepts true grayscale Depth Maps (RGB-D) to physically measure the Z-axis, avoiding AI depth inference inaccuracies and delivering clinical-grade volumetric accuracy.
2. **Explainable AI (XAI) Vector Field Heatmaps:**
   - Rather than just indicating a "magnitude of error," the tool generates directional vector fields directly overlaid on the postoperative images using OpenCV. This allows surgeons to see exactly *which direction* the tissue shifted (e.g., tip rotation, dorsal hump reduction).
3. **The Holistic Outcome Score:**
   - Automatically computes a Surgical Fidelity Index (SFI) based on regional point-to-point Euclidean discrepancy mapping.
   - Blends the objective AI SFI (70% weight) with Patient-Reported Outcome Measures (PROMs) (30% weight) to generate a true holistic measure of clinical success.
4. **Live Multi-Camera Integration:**
   - Built-in `getUserMedia` integration to stream directly from laptop webcams or mobile devices, bypassing the need for manual image uploads in a fast-paced clinical setting.

## Technical Stack
- **Backend:** Python, FastAPI, Uvicorn
- **Computer Vision:** OpenCV, Google MediaPipe FaceLandmarker
- **Math/Alignment:** NumPy (Procrustes Analysis)
- **Frontend:** Vanilla JS, HTML, Custom CSS

<img width="1895" height="1033" alt="image" src="https://github.com/user-attachments/assets/170d5308-c533-4507-b0e0-c79f5d9d4589" />

<img width="1882" height="1037" alt="image" src="https://github.com/user-attachments/assets/f09dc302-d951-43f2-8e7f-13bf4688a31b" />

<img width="1890" height="1002" alt="image" src="https://github.com/user-attachments/assets/d803a210-8239-481a-b51d-a1be78b24025" />


