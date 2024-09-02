import cv2
import face_recognition
import os
import numpy as np
import time

cur_path=r"/home/afrin/Afrin/FACE DETECTION/PICTURES/Afrin.jpg"

img_mine = face_recognition.load_image_file(cur_path)
img_mine = cv2.cvtColor(img_mine,cv2.COLOR_BGR2RGB)

# cv2.imshow('mine_sorce',img_mine)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


faceLoc = face_recognition.face_locations(img_mine)[0]
# print(faceLoc)

cv2.rectangle(img_mine,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
# cv2.imshow('mine_sorce',img_mine)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
encode_mine= face_recognition.face_encodings(img_mine)[0]

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    success, img = video_capture.read()

    # Check if frame is successfully captured
    if not success:
        print("Failed to capture frame from webcam. Exiting...")
        break

    # Resize and convert to RGB for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find all faces and face encodings in the current frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # Compare the face encoding with the known face encoding
        matches = face_recognition.compare_faces([encode_mine], encodeFace)
        faceDis = face_recognition.face_distance([encode_mine], encodeFace)
        matchIndex = np.argmin(faceDis)

        # Check if a match is found and the distance is within threshold
        if matches[matchIndex] and faceDis[matchIndex] < 0.50:
            name = input("Enter your name:")
            print(name)

    # Display the frame with rectangles around detected faces
    for (top, right, bottom, left) in facesCurFrame:
        cv2.rectangle(img, (left * 4, top * 4), (right * 4, bottom * 4), (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', img)

    # Check for user input to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()