import face_recognition
import cv2
import numpy as np

path_src=r"/home/afrin/Afrin/FACE DETECTION/first_pic.webp"
# path_tst=r"/home/afrin/Afrin/FACE DETECTION/second_pic1.webp"
path_tst= r"/home/afrin/Afrin/FACE DETECTION/mohanlal.webp"

img_ekka = face_recognition.load_image_file(path_src)  #LOADING THE IMAGE WITH FACE RECOGNITION
img_ekka = cv2.cvtColor(img_ekka,cv2.COLOR_BGR2RGB)     #CONVERTING TO RGB


imgTestekka = face_recognition.load_image_file(path_tst)
imgTestekka= cv2.cvtColor(imgTestekka,cv2.COLOR_BGR2RGB)

# cv2.imshow('ekka_sorce',img_ekka)
# cv2.imshow('ekka_Test',imgTestekka)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

faceLoc = face_recognition.face_locations(img_ekka)[0]
print(faceLoc)

# cv2.rectangle(img_ekka,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
# cv2.imshow('ekka_sorce',img_ekka)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


faceLocTest = face_recognition.face_locations(imgTestekka)[0]
print(faceLocTest)
# cv2.rectangle(imgTestekka,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
# cv2.imshow('ekka_Test',imgTestekka)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

encode_ekka = face_recognition.face_encodings(img_ekka)[0]
encodeTest = face_recognition.face_encodings(imgTestekka)[0]
results = face_recognition.compare_faces([encode_ekka],encodeTest)
print(results)

results = face_recognition.compare_faces([encode_ekka], encodeTest)
faceDis = face_recognition.face_distance([encodeTest], encodeTest)
print(results, faceDis)
cv2.putText(imgTestekka, f'{results[0]} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),
                2)
#
cv2.imshow('ekka_sorce', img_ekka)
cv2.imshow('ekka_Test', imgTestekka)
cv2.waitKey(0)
cv2.destroyAllWindows()
#
