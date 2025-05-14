import cv2
import numpy as np
import math

def calculate_psnr(video_path1, video_path2):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    
    psnr_total = 0.0
    frame_count = 0
    
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        

        frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))
        

        mse = np.mean((frame1 - frame2) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        
        psnr_total += psnr
        frame_count += 1
    
    cap1.release()
    cap2.release()
    
    return psnr_total / frame_count if frame_count > 0 else 0


psnr_value = calculate_psnr("video_cover/It's_05_10_11_13.mp4", "video_steg/It's_05_10_11_13.mp4")
print(f"Average PSNR: {psnr_value:.2f} dB")
