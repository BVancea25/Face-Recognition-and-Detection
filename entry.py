import cv2
from PIL import Image

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("./classifier.yml")

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(0)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    return faces

while True:
    result, video_frame = video_capture.read()
    if not result:
        break

    faces = detect_bounding_box(video_frame)
    for (x, y, w, h) in faces:
        face_roi = video_frame[y:y+h, x:x+w]
        
        # Convert face region to grayscale
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize face region to a fixed size compatible with the face recognizer model
        resized_face = cv2.resize(gray_face, (100, 100))
        
        # Predict the identity of the face
        label, confidence = face_recognizer.predict(resized_face)
        
        # Display the predicted label on the detected face region
        cv2.putText(video_frame, f"Person: {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.rectangle(video_frame, (x, y), (x+w, y+h), (0, 255, 0), 4)

    cv2.imshow("My Face Detection Project", video_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
