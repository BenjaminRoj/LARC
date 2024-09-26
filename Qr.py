
import cv2
import numpy as np
import os

# Función para cargar las imágenes de referencia
def load_reference_images(path):
    reference_images = {}
    orb = cv2.ORB_create()
    for filename in os.listdir(path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            label = os.path.splitext(filename)[0]
            main_label = label.split(',')[0]  # Extract main label before comma
            img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
            keypoints, descriptors = orb.detectAndCompute(img, None)
            if main_label not in reference_images:
                reference_images[main_label] = []
            reference_images[main_label].append((keypoints, descriptors, img))
    return reference_images

# Función para preprocesar la imagen
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

# Función para encontrar la mejor coincidencia
def find_best_match(image, reference_images, threshold=10):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    if descriptors is None:
        return None, 0, {}, None

    best_match = None
    best_score = 0
    all_matches = {}
    best_homography = None
    for main_label, variations in reference_images.items():
        match_count = 0
        for ref_keypoints, ref_descriptors, ref_image in variations:
            if ref_descriptors is not None and len(ref_descriptors) > 1:
                matches = bf.knnMatch(descriptors, ref_descriptors, k=2)
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                match_count += len(good_matches)
                
                if len(good_matches) > threshold:
                    src_pts = np.float32([ keypoints[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
                    dst_pts = np.float32([ ref_keypoints[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is not None:
                        matches_mask = mask.ravel().tolist()
                        num_matches = np.sum(matches_mask)

                        if num_matches > best_score:
                            best_score = num_matches
                            best_match = main_label
                            best_homography = M
        
        all_matches[main_label] = match_count

    return best_match, best_score, all_matches, best_homography

# Función para dibujar el contorno
def draw_contour(frame, homography, ref_image):
    h, w = ref_image.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, homography)
    frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
    return frame

# Función principal para capturar la cámara en tiempo real y realizar la detección
def main():
    reference_images = load_reference_images(r'C:\Users\HP\Downloads\QRs')  # Cambia esta ruta a donde tienes las imágenes de referencia
    cap = cv2.VideoCapture(0)
    threshold = 50  # Ajusta el umbral según sea necesario
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        preprocessed_frame = preprocess_image(frame)
        best_match, best_score, all_matches, best_homography = find_best_match(preprocessed_frame, reference_images, threshold)
        
        if best_match is not None and best_score > threshold:
            cv2.putText(frame, f'Detected: {best_match}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if best_homography is not None:
                ref_image = reference_images[best_match][0][2]
                frame = draw_contour(frame, best_homography, ref_image)
        else:
            cv2.putText(frame, 'No match detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Crear una imagen de diagnóstico
        diag_image = np.zeros((300, 500, 3), dtype=np.uint8)
        cv2.putText(diag_image, f'Threshold: {threshold}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_offset = 60
        for label, count in all_matches.items():
            cv2.putText(diag_image, f'{label}: {count} matches', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            y_offset += 30
        
        cv2.imshow("Custom Code Scanner", frame)
        cv2.imshow("Diagnostics", diag_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


