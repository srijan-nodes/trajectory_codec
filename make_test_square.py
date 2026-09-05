import cv2
import numpy as np

out = cv2.VideoWriter('test_square.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
for i in range(90):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    x = 100 + i * 2  # Moves 2 pixels per frame right
    y = 100
    frame[y:y+200, x:x+200] = (255, 100, 100) # Blue-ish square
    out.write(frame)
out.release()
print('Created test_square.mp4')
