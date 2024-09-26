import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib

# Cargar el clasificador Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Función para detectar rostros y extraer la región de interés (ROI)
def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Función para preprocesar imágenes
def preprocess_image(img, face):
    x, y, w, h = face
    roi = img[y:y+h, x:x+w]
    roi = cv2.resize(roi, (100, 100))
    roi = roi.flatten()  # Aplanar la imagen para convertirla en un vector
    return roi

# Función para cargar el dataset
def load_dataset(directory):
    X, y = [], []
    labels = []
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if not os.path.isdir(path):
            continue
        labels.append(subdir)
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            faces = detect_faces(img)
            if len(faces) == 0:
                continue
            roi = preprocess_image(img, faces[0])
            X.append(roi)
            y.append(subdir)
    return np.array(X), np.array(y), labels

# Cargar datos de entrenamiento
data_dir = 'dataset'
X, y, labels = load_dataset(data_dir)

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Entrenar clasificador SVM
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X, y_encoded)

# Guardar el modelo y el codificador de etiquetas
joblib.dump(svm_model, 'svm_face_recognition_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Modelo entrenado y guardado exitosamente.")
