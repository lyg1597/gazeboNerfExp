import cv2
import os 

if not os.path.exists('./extracted_splat'):
    os.mkdir('./extracted_splat')

vidcap = cv2.VideoCapture('2024-08-05-20-49-29.mp4')
i = 0
while True:
    success, image = vidcap.read()
    if not success:
        break 
    cv2.imwrite(f'./extracted_splat/frame_{i:05d}.png', image)
    i+=1
