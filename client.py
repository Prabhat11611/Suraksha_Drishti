import cv2
import face_recognition
import requests
import json
from datetime import datetime
import numpy as np
import base64
import time
import logging
import socket
import uuid

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
def get_camera_config():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    unique_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{hostname}:{ip_address}"))
    camera_id = f"CAM-{unique_id[:8]}"  # Use first 8 characters of UUID for brevity
    
    # TODO: Replace this with actual GPS coordinates retrieval
    # For now, we'll use random coordinates for demonstration
    import random
    latitude = random.uniform(-90, 90)
    longitude = random.uniform(-180, 180)
    
    return {
        "CAMERA_ID": camera_id,
        "LATITUDE": latitude,
        "LONGITUDE": longitude
    }

CAMERA_CONFIG = get_camera_config()
SERVER_URL = "http://192.168.81.224:8000"  # Verify this is correct

def get_known_faces():
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            response = requests.get(f"{SERVER_URL}/get_known_faces", timeout=10)
            response.raise_for_status()
            known_faces = response.json()
            known_face_encodings = []
            known_face_names = []
            for face in known_faces:
                encoding = np.frombuffer(base64.b64decode(face['encoding']), dtype=np.float64)
                known_face_encodings.append(encoding)
                known_face_names.append(face['name'])
            logging.info(f"Successfully retrieved {len(known_faces)} known faces")
            return known_face_encodings, known_face_names
        except requests.exceptions.RequestException as e:
            logging.error(f"Error getting known faces (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error("Max retries reached. Unable to get known faces.")
                return [], []

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
        response = requests.post(f"{SERVER_URL}/add_detection", json=face_data, timeout=5)
        response.raise_for_status()
        logging.info(f"Data sent successfully: {response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending data to server: {e}")

def main():
    logging.info(f"Starting face recognition client for {CAMERA_CONFIG['CAMERA_ID']}")
    video_capture = cv2.VideoCapture(0)
    known_face_encodings, known_face_names = get_known_faces()
    
    if not known_face_encodings:
        logging.error("No known faces retrieved. Exiting.")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            logging.error("Failed to capture frame. Exiting.")
            break

        face_locations, face_names = detect_and_recognize_faces(frame, known_face_encodings, known_face_names)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            face_data = {
                "camera_id": CAMERA_CONFIG['CAMERA_ID'],
                "timestamp": datetime.now().isoformat(),
                "person_name": name,
                "bounding_box": {
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "left": left
                },
                "latitude": CAMERA_CONFIG['LATITUDE'],
                "longitude": CAMERA_CONFIG['LONGITUDE']
            }
            send_to_server(face_data)

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()