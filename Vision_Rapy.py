import numpy as np

import cv2

from tflite_runtime.interpreter import Interpreter



# Definir el generador de datos para obtener las etiquetas de las clases

# Aquí no necesitamos `ImageDataGenerator` de Keras ya que solo usaremos la inferencia

# Asegúrate de tener un diccionario con las clases si lo usas en inferencia



# Simulando la carga de etiquetas de clase desde el directorio de entrenamiento

class_labels = ['A', 'B', '1', '2', '3']  # Reemplaza esto con tus clases reales



# Cargar el modelo entrenado (.tflite)

model_path = 'simple_model.tflite'  # Asegúrate de tener el modelo .tflite

interpreter = Interpreter(model_path=model_path)

interpreter.allocate_tensors()



# Obtener detalles de entrada y salida del modelo

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()



def preprocess_image(image):

    image = cv2.resize(image, (150, 150))  # Ajusta esto según el tamaño de entrada de tu modelo

    image = image.astype('float32') / 255

    image = np.expand_dims(image, axis=0)

    return image



def predict(image):

    input_data = preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data



def main():

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        

        predictions = predict(frame)

        best_match = class_labels[np.argmax(predictions)]

        

        cv2.putText(frame, f'Detected: {best_match}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        

        cv2.imshow("Combined Detector", frame)

        

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break

    

    cap.release()

    cv2.destroyAllWindows()



if __name__ == "__main__":

    main()