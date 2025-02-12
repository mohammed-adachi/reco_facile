import cv2
import mediapipe as mp
import face_recognition
import os

# Initialiser MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Charger les visages de référence et leurs encodages
def load_reference_faces(reference_folder):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(reference_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(reference_folder, filename)
            reference_image = face_recognition.load_image_file(image_path)
            reference_encoding = face_recognition.face_encodings(reference_image)

            if reference_encoding:  # Vérifier si un encodage a été trouvé
                known_face_encodings.append(reference_encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])  # Nom du fichier sans extension
                print(f"✅ Visage de référence chargé : {filename}")
            else:
                print(f"❌ Aucun encodage facial trouvé pour {filename}.")

    return known_face_encodings, known_face_names

# Charger les visages de référence
reference_folder = "reference_faces"  # Remplacez par le chemin de votre dossier
known_face_encodings, known_face_names = load_reference_faces(reference_folder)

# Vérifier si des visages de référence ont été chargés
if not known_face_encodings:
    print("❌ Aucun visage de référence valide trouvé. Vérifiez le dossier 'reference_faces'.")
    exit()

# Charger l'image à analyser
image_path = "test.jpg"  # Remplacez par le chemin de votre image
image = cv2.imread(image_path)

if image is None:
    print("❌ Erreur : Impossible de charger l'image.")
    exit()

# Redimensionner l'image pour améliorer les performances
image = cv2.resize(image, (800, 600))

# Convertir l’image en RGB pour MediaPipe
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Détecter les visages avec MediaPipe
results = face_detection.process(image_rgb)

if results.detections:
    face_count = 0  # Compteur pour numéroter les visages
    for detection in results.detections:
        # Récupérer la boîte englobante du visage
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        x1, y1 = int(bboxC.xmin * w), int(bboxC.ymin * h)
        x2, y2 = int((bboxC.xmin + bboxC.width) * w), int((bboxC.ymin + bboxC.height) * h)

        # Ajuster pour éviter les erreurs de découpage
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Découper le visage
        face_crop = image[y1:y2, x1:x2]

        # Vérifier si la découpe est valide
        if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
            print(f"❌ Découpe du visage {face_count} invalide ou trop petite.")
            continue

        # Enregistrer le visage découpé
        face_filename = f"face_{face_count}.jpg"
        cv2.imwrite(face_filename, face_crop)
        print(f"✅ Visage {face_count} enregistré : {face_filename}")

        # Extraire l'encodage facial avec face_recognition
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_encoding = face_recognition.face_encodings(face_crop_rgb)

        if face_encoding:  # Vérifier si un encodage a été trouvé
            print(f"🔑 Encodage facial {face_count} : {face_encoding[0]}")

            # Comparer l'encodage avec les visages de référence
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0], tolerance=0.5)
            name = "Inconnu"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            print(f"👤 Visage {face_count} reconnu comme : {name}")

            # Afficher le nom sur l'image
            cv2.putText(image, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            print(f"❌ Aucun encodage facial trouvé pour le visage {face_count}.")

        # Dessiner la boîte englobante sur l'image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        face_count += 1

# Afficher l'image avec les résultats
cv2.imshow("Résultat de la détection faciale", image)
cv2.waitKey(0)
cv2.destroyAllWindows()