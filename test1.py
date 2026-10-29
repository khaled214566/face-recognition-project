import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(640, 640))

image = cv2.imread(r"E:\yosra cv\A1.jpg")

if image is None:
    raise FileNotFoundError("Check the image path: A1.jpg")

faces = app.get(image)

print("Faces detected:", len(faces))

for face in faces:
    print("Embedding shape:", face.embedding.shape)
    print("Detection confidence:", face.det_score)