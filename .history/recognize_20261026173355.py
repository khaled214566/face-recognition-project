import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(640, 640))


def get_embedding(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")

    faces = app.get(image)

    if len(faces) != 1:
        raise ValueError(
            f"Expected exactly one face in {path}; found {len(faces)}"
        )

    return faces[0].normed_embedding


# Registered person's photo
registered = get_embedding(r"E:\yosra cv\A1.jpg")

# Photo to check
candidate = get_embedding(r"E:\yosra cv\test.jpg")

similarity = float(np.dot(registered, candidate))

print(f"Cosine similarity: {similarity:.3f}")

# Example starting threshold; validate it with your own images
if similarity >= 0.45:
    print("POSSIBLE MATCH")
else:
    print("UNKNOWN PERSON")