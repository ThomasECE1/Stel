"""
API de détection d'émotions basée sur les landmarks faciaux
Détecte 3 émotions : angry, neutral, happy
Utilise MediaPipe pour la détection des points faciaux
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import base64

app = Flask(__name__)
CORS(app)

# Initialiser MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Indices des landmarks importants pour MediaPipe (468 points)
# Bouche
UPPER_LIP_TOP = 13
LOWER_LIP_BOTTOM = 14
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
UPPER_LIP_CENTER = 0
LOWER_LIP_CENTER = 17

# Sourcils
LEFT_EYEBROW_INNER = 107
LEFT_EYEBROW_OUTER = 70
RIGHT_EYEBROW_INNER = 336
RIGHT_EYEBROW_OUTER = 300

# Yeux (pour référence de distance)
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Nez (point de référence)
NOSE_TIP = 4

# Points supplémentaires pour le contour du visage
CHIN = 152
FOREHEAD_CENTER = 10


def calculate_distance(point1, point2):
    """Calcule la distance euclidienne entre deux points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_landmark_point(landmarks, idx, img_width, img_height):
    """Extrait les coordonnées d'un landmark"""
    landmark = landmarks.landmark[idx]
    return (landmark.x * img_width, landmark.y * img_height)


def analyze_facial_features(landmarks, img_width, img_height):
    """
    Analyse les caractéristiques faciales et retourne des métriques
    """
    # Obtenir les points clés
    upper_lip = get_landmark_point(landmarks, UPPER_LIP_TOP, img_width, img_height)
    lower_lip = get_landmark_point(landmarks, LOWER_LIP_BOTTOM, img_width, img_height)
    left_mouth = get_landmark_point(landmarks, LEFT_MOUTH_CORNER, img_width, img_height)
    right_mouth = get_landmark_point(landmarks, RIGHT_MOUTH_CORNER, img_width, img_height)

    left_eyebrow_inner = get_landmark_point(landmarks, LEFT_EYEBROW_INNER, img_width, img_height)
    left_eyebrow_outer = get_landmark_point(landmarks, LEFT_EYEBROW_OUTER, img_width, img_height)
    right_eyebrow_inner = get_landmark_point(landmarks, RIGHT_EYEBROW_INNER, img_width, img_height)
    right_eyebrow_outer = get_landmark_point(landmarks, RIGHT_EYEBROW_OUTER, img_width, img_height)

    left_eye_inner = get_landmark_point(landmarks, LEFT_EYE_INNER, img_width, img_height)
    right_eye_inner = get_landmark_point(landmarks, RIGHT_EYE_INNER, img_width, img_height)

    nose_tip = get_landmark_point(landmarks, NOSE_TIP, img_width, img_height)
    chin = get_landmark_point(landmarks, CHIN, img_width, img_height)

    # Distance de référence (largeur entre les yeux) pour normaliser
    eye_distance = calculate_distance(left_eye_inner, right_eye_inner)

    # --- MÉTRIQUES DE LA BOUCHE ---

    # 1. Ouverture verticale de la bouche (normalisée)
    mouth_open = calculate_distance(upper_lip, lower_lip) / eye_distance

    # 2. Largeur de la bouche (normalisée)
    mouth_width = calculate_distance(left_mouth, right_mouth) / eye_distance

    # 3. Ratio d'aspect de la bouche (largeur / hauteur)
    mouth_aspect_ratio = mouth_width / (mouth_open + 0.001)  # Éviter division par 0

    # 4. Position des coins de la bouche par rapport au centre
    mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
    left_corner_elevation = mouth_center_y - left_mouth[1]  # Positif = coin relevé
    right_corner_elevation = mouth_center_y - right_mouth[1]
    avg_corner_elevation = (left_corner_elevation + right_corner_elevation) / 2
    corner_elevation_normalized = avg_corner_elevation / eye_distance

    # --- MÉTRIQUES DES SOURCILS ---

    # 5. Distance sourcils - yeux (normalisée)
    left_eyebrow_height = calculate_distance(left_eyebrow_inner, left_eye_inner) / eye_distance
    right_eyebrow_height = calculate_distance(right_eyebrow_inner, right_eye_inner) / eye_distance
    avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2

    # 6. Froncement des sourcils (distance entre les sourcils intérieurs)
    eyebrow_inner_distance = calculate_distance(left_eyebrow_inner, right_eyebrow_inner) / eye_distance

    # 7. Inclinaison des sourcils (différence hauteur intérieur vs extérieur)
    left_eyebrow_slope = (left_eyebrow_inner[1] - left_eyebrow_outer[1]) / eye_distance
    right_eyebrow_slope = (right_eyebrow_inner[1] - right_eyebrow_outer[1]) / eye_distance

    return {
        'mouth_open': mouth_open,
        'mouth_width': mouth_width,
        'mouth_aspect_ratio': mouth_aspect_ratio,
        'corner_elevation': corner_elevation_normalized,
        'eyebrow_height': avg_eyebrow_height,
        'eyebrow_inner_distance': eyebrow_inner_distance,
        'left_eyebrow_slope': left_eyebrow_slope,
        'right_eyebrow_slope': right_eyebrow_slope
    }


def classify_emotion(features):
    """
    Classifie l'émotion basée sur les caractéristiques faciales
    Retourne: (emotion, scores_dict)
    """
    scores = {'angry': 0.0, 'neutral': 0.0, 'happy': 0.0}

    # --- RÈGLES POUR HAPPY (sourire) ---
    # Coins de la bouche relevés
    if features['corner_elevation'] > 0.02:
        scores['happy'] += 0.3
    if features['corner_elevation'] > 0.05:
        scores['happy'] += 0.2

    # Bouche large (sourire = bouche plus large)
    if features['mouth_width'] > 1.3:
        scores['happy'] += 0.2
    if features['mouth_width'] > 1.5:
        scores['happy'] += 0.1

    # Ratio d'aspect élevé (bouche large et pas très ouverte)
    if features['mouth_aspect_ratio'] > 4:
        scores['happy'] += 0.15

    # Sourcils légèrement relevés
    if features['eyebrow_height'] > 0.38:
        scores['happy'] += 0.1

    # --- RÈGLES POUR ANGRY (colère) ---
    # Sourcils froncés (rapprochés)
    if features['eyebrow_inner_distance'] < 0.7:
        scores['angry'] += 0.3
    if features['eyebrow_inner_distance'] < 0.5:
        scores['angry'] += 0.2

    # Sourcils inclinés vers le bas au centre (en V)
    avg_slope = (features['left_eyebrow_slope'] + features['right_eyebrow_slope']) / 2
    if avg_slope < -0.02:
        scores['angry'] += 0.2

    # Coins de la bouche abaissés
    if features['corner_elevation'] < -0.02:
        scores['angry'] += 0.2

    # Sourcils bas
    if features['eyebrow_height'] < 0.32:
        scores['angry'] += 0.15

    # --- RÈGLES POUR NEUTRAL ---
    # Bouche ni trop ouverte ni trop large
    if 0.8 < features['mouth_width'] < 1.3:
        scores['neutral'] += 0.2

    # Coins de la bouche au niveau
    if -0.02 < features['corner_elevation'] < 0.02:
        scores['neutral'] += 0.25

    # Sourcils en position normale
    if 0.32 < features['eyebrow_height'] < 0.4:
        scores['neutral'] += 0.2

    # Distance sourcils normale
    if 0.6 < features['eyebrow_inner_distance'] < 0.9:
        scores['neutral'] += 0.2

    # Ajouter un score de base pour neutral (état par défaut)
    scores['neutral'] += 0.15

    # Normaliser les scores pour qu'ils totalisent 1
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    # Déterminer l'émotion dominante
    emotion = max(scores, key=scores.get)

    return emotion, scores


def process_image(image_data):
    """
    Traite une image et retourne l'émotion détectée
    """
    # Décoder l'image base64
    try:
        # Supprimer le préfixe data:image si présent
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return None, None, "Impossible de décoder l'image"

    except Exception as e:
        return None, None, f"Erreur de décodage: {str(e)}"

    # Convertir en RGB pour MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]

    # Détecter les landmarks
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return None, None, "Aucun visage détecté"

    # Prendre le premier visage
    landmarks = results.multi_face_landmarks[0]

    # Analyser les caractéristiques
    features = analyze_facial_features(landmarks, img_width, img_height)

    # Classifier l'émotion
    emotion, scores = classify_emotion(features)

    return emotion, scores, features


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de prédiction d'émotion
    """
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'Image manquante'}), 400

        emotion, scores, features = process_image(data['image'])

        if emotion is None:
            return jsonify({
                'error': features,  # features contient le message d'erreur
                'action': 'MAINTENIR_TEMP'
            }), 200

        # Déterminer l'action de température
        if emotion == 'angry':
            action = "AUGMENTER_TEMP"
        elif emotion == 'happy':
            action = "DIMINUER_TEMP"
        else:  # neutral
            action = "MAINTENIR_TEMP"

        # Convertir scores en pourcentages pour l'affichage
        scores_percent = {k: round(v * 100, 1) for k, v in scores.items()}

        return jsonify({
            'emotion': emotion,
            'scores': scores_percent,
            'action': action,
            'features': {k: round(v, 3) for k, v in features.items()} if isinstance(features, dict) else None
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'action': 'MAINTENIR_TEMP'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Endpoint de vérification de santé"""
    return jsonify({'status': 'ok', 'method': 'landmarks'})


if __name__ == '__main__':
    print("=" * 50)
    print("API Détection d'Émotions par Landmarks Faciaux")
    print("=" * 50)
    print("Méthode: Analyse géométrique des points faciaux")
    print("Émotions: angry, neutral, happy")
    print("Port: 5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
