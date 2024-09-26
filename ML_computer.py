import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definir el generador de datos para obtener las etiquetas de las clases
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Cargar el modelo entrenado
model = load_model('combined_detector_model.h5')

# Definir las etiquetas de las clases
class_labels = list(train_generator.class_indices.keys())

def preprocess_image(image):
    image = cv2.resize(image, (150, 150))
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)
    return image

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        preprocessed_frame = preprocess_image(frame)
        predictions = model.predict(preprocessed_frame)
        best_match = class_labels[np.argmax(predictions)]
        
        cv2.putText(frame, f'Detected: {best_match}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Combined Detector", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
