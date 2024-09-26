import tensorflow as tf

# Definir un modelo simple de Keras
try:
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[150, 150, 3]),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Modelo Keras simple definido correctamente.")
except Exception as e:
    print(f"Error al definir el modelo Keras: {e}")

# Convertir el modelo simple a TensorFlow Lite
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    print("Modelo convertido a TensorFlow Lite correctamente.")
    
    # Guardar el modelo TensorFlow Lite
    with open('simple_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Modelo TensorFlow Lite guardado correctamente.")
except Exception as e:
    print(f"Error al convertir el modelo a TensorFlow Lite: {e}")
