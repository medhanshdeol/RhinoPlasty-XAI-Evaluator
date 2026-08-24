import numpy as np  # type: ignore

class DifferentialAnalyzer:
    def __init__(self):
        # Define the regions based on MediaPipe landmark indices
        # Note: These are rough approximations and need to be refined for clinical use.
        # Dense 80+ points representation for nasal sub-units
        self.regions = {
            'dorsum': [195, 5, 4, 6, 168, 197, 114, 343, 122, 351, 188, 412, 193, 417, 8],
            'tip': [94, 2, 278, 48, 115, 344, 219, 439, 164, 0],
            'alar_base': [129, 358, 279, 49, 98, 327, 218, 438, 235, 455],
            'columella': [2, 164, 94, 97, 326],
            'radix': [168, 6, 8, 9, 107, 336, 108, 337],
            'nasolabial_junction': [164, 167, 393, 165, 392, 0, 11, 12, 13, 14],
        }
        # Combine all sub-units for entire_nose
        self.regions['entire_nose'] = list(set([idx for region in self.regions.values() for idx in region]))

    def compute_discrepancies(self, planned_landmarks, actual_aligned_landmarks):
        """
        Computes 3D distances between planned and actual aligned landmarks.
        Returns an array of discrepancies for all landmarks and their displacement vectors.
        """
        if planned_landmarks.shape != actual_aligned_landmarks.shape:
            raise ValueError("Shapes of planned and actual landmarks must match.")
        
        # Displacement vector (actual - planned)
        vectors = actual_aligned_landmarks - planned_landmarks
        # Calculate euclidean distance for each landmark
        distances = np.linalg.norm(vectors, axis=1)
        return distances, vectors

    def compute_gaussian_score(self, discrepancies, sigma=1.5):
        """
        Maps raw millimeter errors to a smooth 0-1 score using a Gaussian function.
        sigma=1.5 means an error of 1.5mm gets a score of ~0.60, 3mm gets ~0.13.
        """
        return np.exp(-(discrepancies**2) / (2 * (sigma**2)))

    def compute_sfi(self, discrepancies):
        """
        Computes the Plan-vs-Actual Surgical Fidelity Index (SFI) as a percentage (0-100%).
        It aggregates Gaussian scores across the entire nose region.
        """
        nose_indices = self.regions['entire_nose']
        nose_discrepancies = discrepancies[nose_indices]
        scores = self.compute_gaussian_score(nose_discrepancies)
        sfi_percentage = float(np.mean(scores)) * 100
        return sfi_percentage

    def compute_regional_discrepancies(self, discrepancies):
        """
        Aggregates discrepancies into predefined clinical regions using Continuous Gaussian scoring.
        """
        regional_scores = {}
        for region_name, indices in self.regions.items():
            region_dists = discrepancies[indices]
            region_gaussian = self.compute_gaussian_score(region_dists)
            regional_scores[region_name] = {
                'mean_error_mm': float(np.mean(region_dists)),
                'max_error_mm': float(np.max(region_dists)),
                'gaussian_score': float(np.mean(region_gaussian))
            }
        
        # Add SFI overall score
        regional_scores['SFI_overall'] = self.compute_sfi(discrepancies)
        
        return regional_scores
