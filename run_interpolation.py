#!/usr/bin/env python3
"""
Simple RIFE video frame interpolation with M1 GPU (MPS) support
"""

import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm

# Add model path
sys.path.append('train_log')

# Check for MPS availability and set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using M1 GPU (MPS) for acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU (will be slower)")

torch.set_grad_enabled(False)

# Load model
try:
    from train_log.RIFE_HDv3 import Model
    model = Model()
    model.load_model('train_log', -1)
    model.eval()
    model.device()
    
    # Move model to device (MPS/CUDA/CPU)
    if hasattr(model, 'flownet'):
        model.flownet = model.flownet.to(device)
    if hasattr(model, 'contextnet'):
        model.contextnet = model.contextnet.to(device)
    if hasattr(model, 'unet'):
        model.unet = model.unet.to(device)
    
    print("✅ Loaded RIFE v3.x HD model")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

def interpolate_video(input_video, output_video, exp=1, scale=1.0):
    """
    Interpolate frames in a video
    
    Args:
        input_video: Path to input video
        output_video: Path to output video
        exp: Number of times to interpolate (exp=1 means 2x frame rate)
        scale: Scale factor for processing (0.5 for 4K videos)
    """
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {input_video}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Input: {input_video}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Output FPS: {fps * (2 ** exp)}")
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps * (2 ** exp), (width, height))
    
    # Read first frame
    ret, frame1 = cap.read()
    if not ret:
        print("❌ Cannot read first frame")
        return
    
    if scale != 1.0:
        frame1 = cv2.resize(frame1, (width, height))
    
    frame_count = 0
    pbar = tqdm(total=total_frames - 1, desc="Interpolating", ncols=100, colour='green')
    
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        
        if scale != 1.0:
            frame2 = cv2.resize(frame2, (width, height))
        
        # Convert frames to tensors
        img0 = torch.from_numpy(frame1.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        img1 = torch.from_numpy(frame2.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        
        # Move to device (MPS/CUDA/CPU)
        img0 = img0.to(device)
        img1 = img1.to(device)
        
        # Pad to multiple of 64 (required by model)
        n, c, h, w = img0.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
        img0 = torch.nn.functional.pad(img0, padding)
        img1 = torch.nn.functional.pad(img1, padding)
        
        # Write first frame
        out.write(frame1)
        
        # Interpolate frames
        for i in range(2 ** exp - 1):
            timestep = (i + 1) / (2 ** exp)
            
            # Run inference
            with torch.no_grad():
                mid = model.inference(img0, img1, timestep)
            
            # Convert back to numpy
            mid = mid.cpu().numpy().squeeze().transpose(1, 2, 0)
            mid = mid[:h, :w]  # Remove padding
            mid = (mid * 255.0).clip(0, 255).astype(np.uint8)
            
            # Write interpolated frame
            out.write(mid)
        
        # Move to next frame pair
        frame1 = frame2
        frame_count += 1
        pbar.update(1)
    
    # Write last frame
    out.write(frame1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"✅ Output saved to: {output_video}")
    print(f"   Total output frames: {frame_count * (2 ** exp) + 1}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RIFE Video Frame Interpolation with MPS')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', default='output.mp4', help='Output video path')
    parser.add_argument('--exp', type=int, default=1, help='Interpolation times (1=2x, 2=4x, 3=8x)')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor (0.5 for 4K)')
    
    args = parser.parse_args()
    
    interpolate_video(args.input, args.output, args.exp, args.scale)
