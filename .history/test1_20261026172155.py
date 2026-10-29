import face_recognition
import numpy as np
from PIL import Image

# Replace these with your actual image paths
known_path = "A1.jpg"
test_path = "A2.jpg"

# Load images and convert them to standard RGB
known_image = np.array(Image.open(known_path).convert("RGB"))
test_image = np.array(Image.open(test_path).convert("RGB"))

print("Known image:", known_image.shape, known_image.dtype)
print("Test image:", test_image.shape, test_image.dtype)

# Extract face encodings
known_faces = face_recognition.face_encodings(known_image)
test_faces = face_recognition.face_encodings(test_image)

if not known_faces:
    print("No face detected in the registered image.")
elif not test_faces:
    print("No face detected in the test image.")
else:
    distance = face_recognition.face_distance(
        [known_faces[0]], test_faces[0]
    )[0]

    threshold = 0.5

    if distance < threshold:
        print(f"MATCH: Same person (distance: {distance:.3f})")
    else:
        print(f"UNKNOWN: No confident match (distance: {distance:.3f})")