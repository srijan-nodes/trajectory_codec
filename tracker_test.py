import cv2
import numpy as np

def run_dense_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    # Convert to grayscale
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # Create an empty HSV image to represent the motion map
    hsv = np.zeros_like(frame1)
    # Set Saturation to maximum
    hsv[..., 1] = 255

    print("Playing Dense Optical Flow... Press 'q' to quit.")

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 1. THE MAGIC: Farneback Dense Optical Flow
        # Calculates motion for every single pixel
        flow = cv2.calcOpticalFlowFarneback(
            prvs, next_gray, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # 2. Convert Flow to Colors
        # flow[..., 0] is X motion, flow[..., 1] is Y motion
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Angle determines color (Hue)
        hsv[..., 0] = ang * 180 / np.pi / 2
        # Magnitude determines brightness (Value)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the HSV motion map back to BGR so OpenCV can display it
        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Stack the original video and the motion map side-by-side
        combined = np.hstack((frame2, bgr_flow))

        # Shrink it so it fits on most screens
        height, width = combined.shape[:2]
        combined_small = cv2.resize(combined, (width//2, height//2))

        cv2.imshow('Original vs Dense Motion Map', combined_small)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        # Move to next frame
        prvs = next_gray

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_dense_optical_flow("countdown.mp4")