import cv2
import torch
import numpy as np
from src.model import ECGModel
from src.config import *
from src.dsp import (
    safe_savgol, 
    adaptive_clip, 
    get_baseline_mode_high_res, 
    apply_einthoven_smoothing
)

class ECGDigitizer:
    def __init__(self, weights_path):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Initializing ECG Digitizer on {self.device}...")
        
        # Load Architecture
        self.model = ECGModel(pretrained=False).to(self.device)
        
        # Load Weights
        checkpoint = torch.load(weights_path, map_location=self.device)
        # Handle DataParallel keys if model was trained on multi-GPU
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
    def _apply_sharpen(self, img_norm):
        img_uint8 = (img_norm * 255).astype(np.uint8)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
        return cv2.filter2D(img_uint8, -1, kernel).astype(np.float32) / 255.0

    def _apply_clahe(self, img_norm):
        img_uint8 = (img_norm * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    def preprocess_tta(self, image_path):
        """Prepares the 5-Way TTA Batch (Norm, Dark, Bright, Sharp, CLAHE)."""
        # Read and Resize
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop = image[:IMAGE_SIZE[0], :2176] # Crop logic from competition
        crop_norm = crop / 255.0
        
        # Create Augmentations
        t_norm = crop_norm
        t_dark = crop_norm ** 0.8
        t_bright = crop_norm ** 1.2
        t_sharp = self._apply_sharpen(crop_norm)
        t_clahe = self._apply_clahe(crop_norm)
        
        # Stack into Batch [5, 3, H, W]
        batch_np = np.stack([t_norm, t_dark, t_bright, t_sharp, t_clahe])
        batch_tensor = torch.from_numpy(batch_np.transpose(0, 3, 1, 2)).float()
        
        # Resize to Model Input
        batch_tensor = torch.nn.functional.interpolate(
            batch_tensor, size=IMAGE_SIZE, mode='bilinear', align_corners=True
        )
        
        return batch_tensor.to(self.device)

    def extract_signals(self, pixel_map):
        """Converts probability heatmap to 1D voltage arrays."""
        H, W = pixel_map.shape[1], pixel_map.shape[2]
        series = np.zeros((4, W)) # 4 rows (I, II, III, V1...)
        
        for i in range(4):
            # Argmax extraction (finding the peak pixel)
            # Row i, all columns
            heatmap_row = pixel_map[i]
            y_indices = np.argmax(heatmap_row, axis=0) 
            series[i] = y_indices
            
        # Crop to valid time window
        series = series[:, TIME_START:TIME_END]
        return series

    def run_inference(self, image_path):
        # 1. Prepare Batch
        batch = self.preprocess_tta(image_path)
        
        # 2. Model Prediction
        with torch.no_grad():
            output = self.model(batch)
            probs = torch.sigmoid(output).cpu().numpy()
            
        # 3. Ensemble (Weighted Average) - Your Winning Weights
        # Norm(0.40) + Sharp(0.30) + Others(0.10 each)
        pixel_avg = (probs[0] * 0.40) + \
                    (probs[1] * 0.10) + \
                    (probs[2] * 0.10) + \
                    (probs[3] * 0.30) + \
                    (probs[4] * 0.10)
        
        # 4. Signal Extraction
        raw_series_px = self.extract_signals(pixel_avg)
        
        # 5. Conversion & DSP Pipeline
        final_leads = np.zeros_like(raw_series_px)
        
        # Convert Pixel -> Voltage
        for i in range(4):
            # Center zero level
            voltage = (ZERO_LEVELS[i] - raw_series_px[i]) / VOLTAGE_RESOLUTION
            
            # Apply DSP Cleaning
            voltage -= get_baseline_mode_high_res(voltage) # High-Res Centering
            voltage = safe_savgol(voltage, window=7)       # Window 7 Smoothing
            voltage = adaptive_clip(voltage)               # Safety Clip
            
            final_leads[i] = voltage
            
        # 6. Physics Correction (Einthoven's Law)
        # This aligns I, II, III and aVR, aVL, aVF
        final_leads = apply_einthoven_smoothing(final_leads)
        
        return final_leads

if __name__ == "__main__":
    # Test Run
    digitizer = ECGDigitizer(weights_path="weights/iter_0004200.pt")
    print("Model Loaded. Ready for inference.")