import cv2
import numpy as np
import joblib

# Cargar el clasificador Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Cargar el modelo entrenado y el codificador de etiquetas
svm_model = joblib.load('svm_face_recognition_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

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

# Función para reconocimiento facial
def recognize_face(frame):
    faces = detect_faces(frame)
    if len(faces) == 0:
        return frame, "No face detected"
    for face in faces:
        roi = preprocess_image(frame, face)
        roi = np.expand_dims(roi, axis=0)
        yhat_class = svm_model.predict(roi)
        yhat_prob = svm_model.predict_proba(roi)
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        class_label = label_encoder.inverse_transform([class_index])
        
        # Dibujar el rectángulo alrededor del rostro detectado
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        text = f"{class_label[0]} ({class_probability:.2f}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return frame, class_label[0]

# Iniciar captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame, recognized_person = recognize_face(frame)
    
    cv2.imshow('Reconocimiento Facial', frame)
    
    # Presionar 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
