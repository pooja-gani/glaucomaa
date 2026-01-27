import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class GlaucomaMetrics:
    """
    Computes clinical biomarkers from segmentation masks.
    """
    
    @staticmethod
    def compute_vcdr(disc_mask: np.ndarray, cup_mask: np.ndarray) -> float:
        """
        Compute Vertical Cup-to-Disc Ratio (vCDR).
        Masks should be binary (0, 1) or boolean. 
        """
        if disc_mask.sum() == 0:
            return 0.0 # No disc found
            
        # Get bounding boxes
        def get_vertical_diameter(mask):
            rows = np.any(mask, axis=1)
            if not rows.any():
                return 0
            ymin, ymax = np.where(rows)[0][[0, -1]]
            return ymax - ymin

        disc_height = get_vertical_diameter(disc_mask)
        cup_height = get_vertical_diameter(cup_mask)
        
        if disc_height == 0:
            return 0.0
            
        return cup_height / disc_height

    @staticmethod
    def compute_area_ratio(disc_mask: np.ndarray, cup_mask: np.ndarray) -> float:
        """
        Compute Area-based CDR (Cup Area / Disc Area).
        """
        disc_area = np.sum(disc_mask)
        cup_area = np.sum(cup_mask)
        
        if disc_area == 0:
            return 0.0
            
        return cup_area / disc_area

    @staticmethod
    def process_segmentation_output(
        segmentation_map: np.ndarray, 
        disc_curr_idx=1, 
        cup_curr_idx=2
    ) -> dict:
        """
        Process a single segmentation map (H, W) with class indices.
        Returns dictionary of metrics.
        """
        disc_mask = (segmentation_map == disc_curr_idx).astype(np.uint8)
        cup_mask = (segmentation_map == cup_curr_idx).astype(np.uint8)
        
        # Enforce cup inside disc constraint for calculation if needed? 
        # Usually segmentation model learns this, but we can do a logical AND.
        # Ideally, segmentation should be good enough.
        
        vcdr = GlaucomaMetrics.compute_vcdr(disc_mask, cup_mask)
        area_cdr = GlaucomaMetrics.compute_area_ratio(disc_mask, cup_mask)
        
        return {
            "vCDR": vcdr,
            "Area_CDR": area_cdr,
            "Disc_Area_Px": int(np.sum(disc_mask)),
            "Cup_Area_Px": int(np.sum(cup_mask))
        }

    @staticmethod
    def dice_coeff(pred, target, smooth=1.):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
