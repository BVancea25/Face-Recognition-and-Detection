import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

directory=Path("./poze")

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

faces=[]
ids=[]

for file in directory.iterdir():
    if file.is_file():
        print(file.name)
        img = cv2.resize(cv2.imread(str(file)),(500,500))
        id=int(file.name.split('_')[0])

        gray_scale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        face = face_classifier.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=7, minSize=(65, 65))
        
        for (x, y, w, h) in face:
            face_roi = gray_scale[y:y+h, x:x+w]
            img_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                        
            plt.figure(figsize=(20,10))
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.show()
            face_roi = cv2.resize(face_roi, (200, 200))
            faces.append(face_roi)
            ids.append(id)
            




face_recognizer.train(faces,np.array(ids))

face_recognizer.write('./classifier.yml')



