"""
API de détection d'émotions basée sur OpenCV + dlib
Détecte 3 émotions : angry, neutral, happy
Utilise dlib pour la détection des 68 points faciaux
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import dlib
import os

app = Flask(__name__)
CORS(app)

# Télécharger le modèle de landmarks si nécessaire
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")

# Initialiser dlib
detector = dlib.get_frontal_face_detector()

if os.path.exists(PREDICTOR_PATH):
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    USE_LANDMARKS = True
    print("✓ Modèle de landmarks chargé")
else:
    predictor = None
    USE_LANDMARKS = False
    print("⚠ Modèle de landmarks non trouvé")
    print(f"  Téléchargez-le depuis:")
    print(f"  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print(f"  Décompressez et placez dans: {BASE_DIR}")
    print("  Mode simplifié activé (détection basique)")


def calculate_distance(p1, p2):
    """Calcule la distance euclidienne entre deux points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_landmarks(shape):
    """Extrait les 68 landmarks comme liste de tuples (x, y)"""
    return [(shape.part(i).x, shape.part(i).y) for i in range(68)]


def analyze_face_landmarks(landmarks):
    """
    Analyse les 68 landmarks faciaux dlib

    Points importants:
    - Sourcils: 17-21 (gauche), 22-26 (droit)
    - Yeux: 36-41 (gauche), 42-47 (droit)
    - Nez: 27-35
    - Bouche: 48-67
    """

    # Points de la bouche
    mouth_left = landmarks[48]      # Coin gauche
    mouth_right = landmarks[54]     # Coin droit
    mouth_top = landmarks[51]       # Centre lèvre supérieure
    mouth_bottom = landmarks[57]    # Centre lèvre inférieure
    upper_lip_top = landmarks[62]   # Intérieur lèvre supérieure
    lower_lip_bottom = landmarks[66] # Intérieur lèvre inférieure

    # Points des sourcils
    left_eyebrow_inner = landmarks[21]
    left_eyebrow_outer = landmarks[17]
    right_eyebrow_inner = landmarks[22]
    right_eyebrow_outer = landmarks[26]

    # Points des yeux
    left_eye_inner = landmarks[39]
    left_eye_outer = landmarks[36]
    right_eye_inner = landmarks[42]
    right_eye_outer = landmarks[45]

    # Nez
    nose_tip = landmarks[30]

    # Distance de référence (largeur entre les yeux)
    eye_distance = calculate_distance(left_eye_inner, right_eye_inner)

    if eye_distance < 1:
        eye_distance = 1  # Éviter division par zéro

    # === MÉTRIQUES DE LA BOUCHE ===

    # Ouverture de la bouche
    mouth_open = calculate_distance(upper_lip_top, lower_lip_bottom) / eye_distance

    # Largeur de la bouche
    mouth_width = calculate_distance(mouth_left, mouth_right) / eye_distance

    # Élévation des coins de la bouche
    mouth_center_y = (mouth_top[1] + mouth_bottom[1]) / 2
    left_corner_elevation = mouth_center_y - mouth_left[1]
    right_corner_elevation = mouth_center_y - mouth_right[1]
    avg_corner_elevation = (left_corner_elevation + right_corner_elevation) / (2 * eye_distance)

    # === MÉTRIQUES DES SOURCILS ===

    # Hauteur des sourcils par rapport aux yeux
    left_eyebrow_height = calculate_distance(left_eyebrow_inner, left_eye_inner) / eye_distance
    right_eyebrow_height = calculate_distance(right_eyebrow_inner, right_eye_inner) / eye_distance
    avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2

    # Distance entre les sourcils (froncement)
    eyebrow_inner_distance = calculate_distance(left_eyebrow_inner, right_eyebrow_inner) / eye_distance

    # Inclinaison des sourcils
    left_eyebrow_slope = (left_eyebrow_inner[1] - left_eyebrow_outer[1]) / eye_distance
    right_eyebrow_slope = (right_eyebrow_inner[1] - right_eyebrow_outer[1]) / eye_distance

    return {
        'mouth_open': mouth_open,
        'mouth_width': mouth_width,
        'corner_elevation': avg_corner_elevation,
        'eyebrow_height': avg_eyebrow_height,
        'eyebrow_inner_distance': eyebrow_inner_distance,
        'left_eyebrow_slope': left_eyebrow_slope,
        'right_eyebrow_slope': right_eyebrow_slope
    }


def classify_emotion(features):
    """
    Classifie l'émotion en 3 catégories: angry, neutral, happy
    """
    scores = {'angry': 0.0, 'neutral': 0.0, 'happy': 0.0}

    # === RÈGLES POUR HAPPY ===

    # Coins de la bouche relevés (sourire)
    if features['corner_elevation'] > 0.015:
        scores['happy'] += 0.35
    if features['corner_elevation'] > 0.03:
        scores['happy'] += 0.2

    # Bouche large
    if features['mouth_width'] > 1.4:
        scores['happy'] += 0.2
    if features['mouth_width'] > 1.6:
        scores['happy'] += 0.15

    # Sourcils légèrement relevés
    if features['eyebrow_height'] > 0.35:
        scores['happy'] += 0.1

    # === RÈGLES POUR ANGRY ===

    # Sourcils froncés (rapprochés)
    if features['eyebrow_inner_distance'] < 0.8:
        scores['angry'] += 0.25
    if features['eyebrow_inner_distance'] < 0.6:
        scores['angry'] += 0.2

    # Sourcils inclinés vers le centre (en V)
    avg_slope = (features['left_eyebrow_slope'] + features['right_eyebrow_slope']) / 2
    if avg_slope < -0.015:
        scores['angry'] += 0.2

    # Coins de la bouche abaissés
    if features['corner_elevation'] < -0.01:
        scores['angry'] += 0.2

    # Sourcils bas
    if features['eyebrow_height'] < 0.28:
        scores['angry'] += 0.15

    # === RÈGLES POUR NEUTRAL ===

    # Position normale de la bouche
    if -0.015 < features['corner_elevation'] < 0.015:
        scores['neutral'] += 0.3

    # Largeur normale de la bouche
    if 1.1 < features['mouth_width'] < 1.5:
        scores['neutral'] += 0.2

    # Sourcils en position normale
    if 0.28 < features['eyebrow_height'] < 0.38:
        scores['neutral'] += 0.2

    # Base neutre
    scores['neutral'] += 0.15

    # Normaliser
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    emotion = max(scores, key=scores.get)
    return emotion, scores


def analyze_face_simple(face_roi):
    """
    Analyse simplifiée basée sur les proportions du visage
    (utilisé si le modèle de landmarks n'est pas disponible)
    """
    # Cette méthode est très basique et moins précise
    h, w = face_roi.shape[:2]

    # Diviser le visage en régions
    upper_third = face_roi[0:h//3, :]
    middle_third = face_roi[h//3:2*h//3, :]
    lower_third = face_roi[2*h//3:, :]

    # Calculer l'intensité moyenne de chaque région
    upper_intensity = np.mean(upper_third)
    lower_intensity = np.mean(lower_third)

    # Détection basique basée sur le contraste
    scores = {'angry': 0.33, 'neutral': 0.34, 'happy': 0.33}

    return 'neutral', scores, None


def process_image(image_data):
    """
    Traite une image et retourne l'émotion détectée
    """
    try:
        # Décoder l'image base64
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return None, None, None, "Impossible de décoder l'image"

    except Exception as e:
        return None, None, None, f"Erreur de décodage: {str(e)}"

    # Convertir en niveaux de gris pour dlib
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = detector(gray, 0)

    if len(faces) == 0:
        return None, None, None, "Aucun visage détecté"

    # Prendre le premier visage
    face = faces[0]

    if USE_LANDMARKS and predictor is not None:
        # Détecter les landmarks
        shape = predictor(gray, face)
        landmarks = get_landmarks(shape)

        # Analyser les caractéristiques
        features = analyze_face_landmarks(landmarks)

        # Classifier
        emotion, scores = classify_emotion(features)

        return emotion, scores, features, None
    else:
        # Mode simplifié
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_roi = gray[max(0,y):y+h, max(0,x):x+w]
        emotion, scores, features = analyze_face_simple(face_roi)
        return emotion, scores, features, None


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'Image manquante'}), 400

        emotion, scores, features, error = process_image(data['image'])

        if error:
            return jsonify({
                'error': error,
                'action': 'MAINTENIR_TEMP'
            }), 200

        # Déterminer l'action
        if emotion == 'angry':
            action = "AUGMENTER_TEMP"
        elif emotion == 'happy':
            action = "DIMINUER_TEMP"
        else:
            action = "MAINTENIR_TEMP"

        scores_percent = {k: round(v * 100, 1) for k, v in scores.items()}

        return jsonify({
            'emotion': emotion,
            'scores': scores_percent,
            'action': action,
            'features': {k: round(v, 4) for k, v in features.items()} if features else None,
            'method': 'landmarks' if USE_LANDMARKS else 'simple'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'action': 'MAINTENIR_TEMP'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'method': 'dlib_landmarks' if USE_LANDMARKS else 'simple',
        'landmarks_available': USE_LANDMARKS
    })


if __name__ == '__main__':
    print("=" * 50)
    print("API Détection d'Émotions - OpenCV + dlib")
    print("=" * 50)
    print(f"Landmarks: {'Activé' if USE_LANDMARKS else 'Désactivé'}")
    print("Émotions: angry, neutral, happy")
    print("Port: 5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
