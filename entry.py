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
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 6, minSize=(30, 30))
    return faces

def start_stream():
    
    while True:
        result, video_frame = video_capture.read()
        if not result:
            break
        else:
            faces = detect_bounding_box(video_frame)
            for (x, y, w, h) in faces:
                face_roi = video_frame[y:y+h, x:x+w]
                
                # Convert face region to grayscale
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                
                # Resize face region to a fixed size compatible with the face recognizer model
                resized_face = cv2.resize(gray_face, (200, 200))
                
                # Predict the identity of the face
                label, confidence = face_recognizer.predict(resized_face)
                
                # Display the predicted label on the detected face region
                cv2.putText(video_frame, f"Person: {label} {int(confidence)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.rectangle(video_frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
            ret, buffer=cv2.imencode('.jpg',video_frame)
            video_frame=buffer.tobytes()    
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + video_frame + b'\r\n')
            


