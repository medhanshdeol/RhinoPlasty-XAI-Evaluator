import sys
import os
import cv2  # type: ignore

from preprocess.mediapipe_extractor import FaceLandmarkExtractor
from analysis.differential_analyzer import DifferentialAnalyzer
from xai.anomaly_visualizer import generate_discrepancy_heatmap

def run_test():
    plan_path = "test_images/dummy_pre.jpg"
    actual_path = "test_images/dummy_post.jpg"
    
    print(f"Loading plan: {plan_path}")
    print(f"Loading actual: {actual_path}")
    
    extractor = FaceLandmarkExtractor()
    analyzer = DifferentialAnalyzer()
    
    print("Extracting landmarks...")
    plan_res = extractor.extract_landmarks(plan_path)
    actual_res = extractor.extract_landmarks(actual_path)
    
    if plan_res is None:
        print(f"Failed to detect face in {plan_path}")
        return
    if actual_res is None:
        print(f"Failed to detect face in {actual_path}")
        return
        
    plan_landmarks_mm, _ = plan_res
    actual_landmarks_mm, actual_landmarks_norm = actual_res
        
    print("Aligning actual to plan using Procrustes analysis...")
    actual_aligned = extractor.procrustes_align(actual_landmarks_mm, plan_landmarks_mm)
    
    print("Computing point-by-point 3D discrepancies...")
    discrepancies, vectors = analyzer.compute_discrepancies(plan_landmarks_mm, actual_aligned)
    
    print("Calculating regional clinical scores...")
    regional_scores = analyzer.compute_regional_discrepancies(discrepancies)
    
    print("\n=== Regional Discrepancy Scores ===")
    for region, scores in regional_scores.items():
        if region == 'SFI_overall':
            print(f"{region.upper()}: {scores:.2f}%")
        else:
            print(f"{region.upper()}:")
            print(f"  Mean Error: {scores['mean_error_mm']:.4f} mm")
            print(f"  Max Error:  {scores['max_error_mm']:.4f} mm")
            print(f"  Score (Gaussian): {scores['gaussian_score']:.4f}")
        
    heatmap_output = "test_images/heatmap_output.jpg"
    print(f"\nGenerating visual heatmap overlay...")
    generate_discrepancy_heatmap(actual_path, actual_landmarks_norm, discrepancies, vectors, heatmap_output)
    print(f"Heatmap saved to {heatmap_output}")
    print("Test completed successfully.")

if __name__ == "__main__":
    run_test()
