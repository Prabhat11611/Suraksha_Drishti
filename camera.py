import cv2
import face_recognition
import requests
import json
from datetime import datetime
import numpy as np
import base64

# Configuration
CAMERA_ID = "CAM1"
SERVER_URL = "http://192.120.26.205:8000"

def get_known_faces():
    response = requests.get(f"{SERVER_URL}/get_known_faces")
    known_faces = response.json()
    known_face_encodings = []
    known_face_names = []
    for face in known_faces:
        encoding = np.frombuffer(base64.b64decode(face['encoding']), dtype=np.float64)
        known_face_encodings.append(encoding)
        known_face_names.append(face['name'])
    return known_face_encodings, known_face_names

def detect_and_recognize_faces(frame, known_face_encodings, known_face_names):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    return face_locations, face_names

def send_to_server(face_data):
    try:
        response = requests.post(f"{SERVER_URL}/add_detection", json=face_data)
        response.raise_for_status()
        print(f"Data sent successfully: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending data to server: {e}")

def main():
    video_capture = cv2.VideoCapture(0)

    known_face_encodings, known_face_names = get_known_faces()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        face_locations, face_names = detect_and_recognize_faces(frame, known_face_encodings, known_face_names)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            face_data = {
                "camera_id": CAMERA_ID,
                "timestamp": datetime.now().isoformat(),
                "person_name": name,
                "bounding_box": {
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "left": left
                }
            }
            send_to_server(face_data)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
