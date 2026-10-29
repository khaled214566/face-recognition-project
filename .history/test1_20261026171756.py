import face_recognition
import numpy as np

# Register a person in your database
known_image = face_recognition.load_image_file(
    "A1"
)
known_faces = face_recognition.face_encodings(known_image)

if not known_faces:
    raise ValueError("No face detected in the registration image")

person_database = [{
    "person_id": "person_001",
    "name": "Person A",
    "encoding": known_faces[0],
}]

# Recognize a new image
unknown_image = face_recognition.load_image_file(
    "A2"
)
unknown_faces = face_recognition.face_encodings(unknown_image)

for face in unknown_faces:
    encodings = [person["encoding"] for person in person_database]

    distances = face_recognition.face_distance(encodings, face)
    best_index = int(np.argmin(distances))

    if distances[best_index] < 0.5:  # Example threshold; validate it
        print("Recognized:", person_database[best_index]["name"])
    else:
        print("UNKNOWN PERSON")