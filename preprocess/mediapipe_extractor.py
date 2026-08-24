import cv2  # type: ignore
import mediapipe as mp  # type: ignore
from mediapipe.tasks import python  # type: ignore
from mediapipe.tasks.python import vision  # type: ignore
import numpy as np  # type: ignore
import math
import os

class FaceLandmarkExtractor:
    def __init__(self, static_image_mode=True, max_num_faces=1):
        # Path to the task file relative to this script
        model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe model not found at {model_path}. Please download it first.")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_num_faces
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def extract_landmarks(self, image_path, depth_map_path=None):
        """Extracts 3D landmarks from an image path, optionally overriding Z with true depth."""
        try:
            image_cv = cv2.imread(image_path)
            if image_cv is None:
                raise ValueError(f"OpenCV could not read image at {image_path}")
            height, width, _ = image_cv.shape
            # Convert to MediaPipe format
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        except Exception as e:
            raise ValueError(f"Could not read image at {image_path}: {e}")
            
        detection_result = self.detector.detect(image_mp)
        
        if not detection_result.face_landmarks:
            return None
        
        # We assume one face per image for this pipeline
        landmarks = detection_result.face_landmarks[0]
        
        # Load depth map if provided
        depth_data = None
        if depth_map_path and os.path.exists(depth_map_path):
            try:
                depth_data = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
                if depth_data is not None:
                    depth_data = cv2.resize(depth_data, (width, height))
            except Exception as e:
                print(f"Warning: Could not read depth map at {depth_map_path}: {e}")
        
        # Convert landmarks to a numpy array (N, 3)
        num_landmarks = len(landmarks)
        coords_norm = np.zeros((num_landmarks, 3))
        for i, lm in enumerate(landmarks):
            z_val = lm.z
            if depth_data is not None:
                px = int(np.clip(lm.x * width, 0, width - 1))
                py = int(np.clip(lm.y * height, 0, height - 1))
                # Map 0-255 intensity to mm depth (assume ~100mm variance for demo)
                z_val = (depth_data[py, px] / 255.0) * 100.0
                
            coords_norm[i] = [lm.x, lm.y, z_val]
            
        # Metric Calibration using Anchor B: Intercanthal Distance (32.0 mm)
        # Landmarks: Right Endocanthion (133), Left Endocanthion (362)
        p133 = coords_norm[133]
        p362 = coords_norm[362]
        d_intercanthal = np.linalg.norm(p133 - p362)
        coords = coords_norm.copy()
        if d_intercanthal > 0:
            scale_mm = 32.0 / d_intercanthal
            coords = coords_norm * scale_mm
            
        return coords, coords_norm

    @staticmethod
    def procrustes_align(source, target, scale=False, anchor_indices=None):
        """
        Aligns source landmarks to target landmarks using Procrustes analysis.
        Both source and target should be (N, 3) numpy arrays.
        Returns the aligned source array.
        """
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape")
            
        if anchor_indices is None:
            # Default to cranial anchors: 
            # Right/Left inner canthi (133, 362), Right/Left outer canthi (33, 263), Nasion (168)
            anchor_indices = [33, 133, 168, 263, 362]
            
        # Use only anchors for finding transformation
        source_anchors = source[anchor_indices]
        target_anchors = target[anchor_indices]
            
        # Translate to origin
        mu_source = source_anchors.mean(axis=0)
        mu_target = target_anchors.mean(axis=0)
        
        source_zerocentered = source_anchors - mu_source
        target_zerocentered = target_anchors - mu_target
        
        # Scale to unit variance (We default scale=False since we already calibrated to mm)
        if scale:
            norm_source = np.linalg.norm(source_zerocentered)
            norm_target = np.linalg.norm(target_zerocentered)
            if norm_source > 0 and norm_target > 0:
                source_zerocentered /= norm_source
                target_zerocentered /= norm_target
                scale_factor = norm_target / norm_source
            else:
                scale_factor = 1.0
        else:
            scale_factor = 1.0
            
        # Rotation using SVD
        U, S, Vt = np.linalg.svd(np.dot(target_zerocentered.T, source_zerocentered))
        R = np.dot(U, Vt)
        
        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(U, Vt)
            
        # Apply transformation to the ENTIRE source
        source_aligned = np.dot(source - mu_source, R.T) * scale_factor + mu_target
        
        return source_aligned

    def calculate_nasolabial_angle(self, landmarks):
        """
        Calculates a proxy for the nasolabial angle using specific landmarks.
        Using MediaPipe Face Mesh landmarks:
        - Subnasale (under the nose): ~164 or 2
        - Columella (base of nose tip): ~94
        - Upper lip margin: ~0 or 11
        This is a simplified 2D/3D proxy.
        """
        if landmarks is None:
            return None
        
        # Define indices (these are approximate, need precise tuning based on MediaPipe map)
        pt_tip = landmarks[94]
        pt_subnasale = landmarks[164]
        pt_lip = landmarks[0]
        
        # Vector from subnasale to tip
        vec1 = pt_tip - pt_subnasale
        # Vector from subnasale to lip
        vec2 = pt_lip - pt_subnasale
        
        # Calculate angle
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Prevent division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        cos_theta = dot_product / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        angle_rad = math.acos(cos_theta)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg

    def calculate_asymmetry_index(self, landmarks):
        if landmarks is None:
            return None

        left_indices = [234, 93, 132, 58]
        right_indices = [454, 323, 361, 288]
        
        asymmetry = 0.0
        for l_idx, r_idx in zip(left_indices, right_indices):
            
            mid_pt = landmarks[4] 
            dist_l = np.linalg.norm(landmarks[l_idx] - mid_pt)
            dist_r = np.linalg.norm(landmarks[r_idx] - mid_pt)
            asymmetry += abs(dist_l - dist_r)
            
        return asymmetry

    def extract_geometric_features(self, image_path):
        res = self.extract_landmarks(image_path)
        if res is None:
            return None
        landmarks, _ = res
            
        angle = self.calculate_nasolabial_angle(landmarks)
        asym = self.calculate_asymmetry_index(landmarks)
        return np.array([angle, asym])

if __name__ == "__main__":
    print("MediaPipe Extractor loaded.")

