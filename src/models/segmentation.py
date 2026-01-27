import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class GlaucomaSegmentationModel(nn.Module):
    """
    SegFormer model for Optic Disc and Cup segmentation.
    Pretrained model: pamixsun/segformer_for_optic_disc_cup_segmentation
    """
    def __init__(self, classes=None, **kwargs):
        super().__init__()
        # Ignoring other arguments to maintain compatibility with existing calls
        # Loading the specific pretrained model requested
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "pamixsun/segformer_for_optic_disc_cup_segmentation"
        )
        
    def forward(self, x):
        # SegFormer expects 'pixel_values'
        outputs = self.model(pixel_values=x)
        
        # Upsample logits to match input size
        # SegFormer outputs are typically 1/4th of the input size
        logits = nn.functional.interpolate(
            outputs.logits, 
            size=x.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        return logits

    def predict_step(self, x):
        """
        Inference step returning class indices and probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            # Assuming softmax for multiclass
            probs = torch.softmax(logits, dim=1)
            # Take argmax for hard prediction
            preds = torch.argmax(probs, dim=1) 
        return preds, probs
