"""
API de détection d'émotions SIMPLE - OpenCV uniquement
Détecte 3 émotions : angry, neutral, happy
Utilise Haar Cascades pour la détection du visage et de la bouche
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)

# Charger les classificateurs Haar
cv2_data_path = cv2.data.haarcascades

face_cascade = cv2.CascadeClassifier(cv2_data_path + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2_data_path + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2_data_path + 'haarcascade_eye.xml')

print("✓ Classificateurs Haar chargés")


def detect_smile(face_roi_gray):
    """
    Détecte un sourire dans la région du visage
    Retourne le score de sourire (nombre et taille des détections)
    """
    h, w = face_roi_gray.shape
    mouth_region = face_roi_gray[int(h*0.5):, :]

    smiles = smile_cascade.detectMultiScale(
        mouth_region,
        scaleFactor=1.7,
        minNeighbors=20,
        minSize=(25, 25)
    )

    if len(smiles) > 0:
        largest_smile = max(smiles, key=lambda s: s[2] * s[3])
        smile_area = largest_smile[2] * largest_smile[3]
        mouth_area = mouth_region.shape[0] * mouth_region.shape[1]
        smile_ratio = smile_area / mouth_area
        return min(smile_ratio * 5, 1.0)
    return 0.0


def detect_frown(face_roi_gray):
    """
    Détecte un froncement de sourcils avec plusieurs méthodes combinées
    """
    h, w = face_roi_gray.shape

    # Zone centrale entre les sourcils (glabelle) - c'est là que les rides apparaissent
    glabella_region = face_roi_gray[int(h*0.15):int(h*0.35), int(w*0.35):int(w*0.65)]

    # Zone des sourcils gauche et droite
    left_brow = face_roi_gray[int(h*0.15):int(h*0.35), int(w*0.15):int(w*0.40)]
    right_brow = face_roi_gray[int(h*0.15):int(h*0.35), int(w*0.60):int(w*0.85)]

    frown_indicators = 0.0

    # 1. Rides verticales dans la glabelle (entre les sourcils)
    if glabella_region.size > 0:
        # Gradient vertical pour détecter les rides verticales
        sobel_v = cv2.Sobel(glabella_region, cv2.CV_64F, 1, 0, ksize=3)
        vertical_lines = np.mean(np.abs(sobel_v))
        frown_indicators += min(vertical_lines / 30, 0.4)

    # 2. Contraste élevé dans la zone glabelle (rides = fort contraste)
    if glabella_region.size > 0:
        glabella_std = np.std(glabella_region)
        if glabella_std > 35:
            frown_indicators += 0.3
        elif glabella_std > 25:
            frown_indicators += 0.2
        elif glabella_std > 15:
            frown_indicators += 0.1

    # 3. Assombrissement de la zone glabelle (ombre des sourcils froncés)
    if glabella_region.size > 0 and left_brow.size > 0 and right_brow.size > 0:
        glabella_mean = np.mean(glabella_region)
        brows_mean = (np.mean(left_brow) + np.mean(right_brow)) / 2
        # Si la glabelle est plus sombre que les côtés = sourcils rapprochés
        if glabella_mean < brows_mean - 10:
            frown_indicators += 0.3
        elif glabella_mean < brows_mean - 5:
            frown_indicators += 0.15

    # 4. Texture rugueuse dans la région des sourcils
    if glabella_region.size > 0:
        laplacian = cv2.Laplacian(glabella_region, cv2.CV_64F)
        texture_score = np.var(laplacian)
        if texture_score > 500:
            frown_indicators += 0.2
        elif texture_score > 300:
            frown_indicators += 0.1

    return min(frown_indicators, 1.0)


def analyze_mouth_shape(face_roi_gray):
    """
    Analyse la forme de la bouche
    Retourne: aspect_ratio, mouth_curve
    """
    h, w = face_roi_gray.shape
    mouth_region = face_roi_gray[int(h*0.6):int(h*0.9), int(w*0.2):int(w*0.8)]
    mh, mw = mouth_region.shape

    # Aspect ratio
    blurred = cv2.GaussianBlur(mouth_region, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    aspect_ratio = 2.0
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        if ch > 0:
            aspect_ratio = cw / ch

    # Courbure de la bouche
    left_zone = mouth_region[:, :int(mw*0.3)]
    center_zone = mouth_region[:, int(mw*0.35):int(mw*0.65)]
    right_zone = mouth_region[:, int(mw*0.7):]

    def find_lip_y(zone):
        if zone.size == 0:
            return zone.shape[0] / 2 if zone.shape[0] > 0 else 0
        threshold = np.percentile(zone, 30)
        dark_pixels = np.where(zone < threshold)
        if len(dark_pixels[0]) > 10:
            return np.mean(dark_pixels[0])
        return zone.shape[0] / 2

    left_y = find_lip_y(left_zone)
    center_y = find_lip_y(center_zone)
    right_y = find_lip_y(right_zone)

    avg_corners_y = (left_y + right_y) / 2
    mouth_curve = (avg_corners_y - center_y) / mh if mh > 0 else 0

    return aspect_ratio, mouth_curve


def analyze_face_intensity(face_roi_gray):
    """
    Analyse l'intensité et le contraste
    """
    h, w = face_roi_gray.shape

    brow_region = face_roi_gray[int(h*0.1):int(h*0.3), :]
    brow_std = np.std(brow_region)

    mouth_region = face_roi_gray[int(h*0.6):, :]
    mouth_std = np.std(mouth_region)

    return {
        'brow_contrast': brow_std,
        'mouth_contrast': mouth_std
    }


def classify_emotion(smile_score, frown_score, mouth_ratio, intensity, mouth_curve=0):
    """
    Classifie l'émotion en combinant tous les indicateurs
    - HAPPY: sourire détecté, coins de bouche relevés
    - ANGRY: sourcils froncés (critère principal)
    - NEUTRAL: expression neutre
    """
    scores = {'angry': 0.0, 'neutral': 0.0, 'happy': 0.0}

    # === HAPPY ===
    # Sourire détecté par Haar Cascade
    scores['happy'] += smile_score * 0.5

    # Coins de bouche relevés
    if mouth_curve > 0.08:
        scores['happy'] += 0.3
    elif mouth_curve > 0.04:
        scores['happy'] += 0.2
    elif mouth_curve > 0.02:
        scores['happy'] += 0.1

    # Bouche large (sourire)
    if mouth_ratio > 3.0:
        scores['happy'] += 0.15
    if mouth_ratio > 4.0:
        scores['happy'] += 0.1

    # === ANGRY ===
    # SOURCILS FRONCÉS = critère principal pour angry
    if frown_score > 0.5:
        scores['angry'] += 0.65
    elif frown_score > 0.4:
        scores['angry'] += 0.5
    elif frown_score > 0.35:
        scores['angry'] += 0.35
    elif frown_score > 0.3:
        scores['angry'] += 0.2

    # Contraste élevé dans les sourcils (rides de froncement)
    if intensity['brow_contrast'] > 45:
        scores['angry'] += 0.1

    # Pas de sourire renforce angry
    if smile_score < 0.1 and frown_score > 0.35:
        scores['angry'] += 0.1

    # === NEUTRAL ===
    # Pas de sourire significatif
    if smile_score < 0.15:
        scores['neutral'] += 0.15

    # Sourcils détendus (pas froncés)
    if frown_score < 0.3:
        scores['neutral'] += 0.25
    elif frown_score < 0.38:
        scores['neutral'] += 0.1

    # Bouche droite
    if -0.03 < mouth_curve < 0.03:
        scores['neutral'] += 0.1

    # Base neutre
    scores['neutral'] += 0.12

    # Normaliser
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    emotion = max(scores, key=scores.get)
    return emotion, scores


def process_image(image_data):
    """
    Traite une image et retourne l'émotion
    """
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return None, None, None, "Impossible de décoder l'image"

    except Exception as e:
        return None, None, None, f"Erreur: {str(e)}"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) == 0:
        return None, None, None, "Aucun visage détecté"

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y+h, x:x+w]

    smile_score = detect_smile(face_roi)
    frown_score = detect_frown(face_roi)
    mouth_ratio, mouth_curve = analyze_mouth_shape(face_roi)
    intensity = analyze_face_intensity(face_roi)

    emotion, scores = classify_emotion(smile_score, frown_score, mouth_ratio, intensity, mouth_curve)

    # Debug: afficher tous les scores pour ajuster les seuils
    print(f"[DEBUG] smile={smile_score:.2f} frown={frown_score:.2f} | ANGRY={scores['angry']*100:.0f}% NEUTRAL={scores['neutral']*100:.0f}% HAPPY={scores['happy']*100:.0f}% → {emotion.upper()}")

    features = {
        'smile_score': smile_score,
        'frown_score': frown_score,
        'mouth_ratio': mouth_ratio,
        'mouth_curve': mouth_curve,
        'brow_contrast': intensity['brow_contrast'],
        'mouth_contrast': intensity['mouth_contrast']
    }

    return emotion, scores, features, None


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    try:
        image_data = None

        if 'image' in request.files:
            file = request.files['image']
            image_data = base64.b64encode(file.read()).decode('utf-8')
        elif request.is_json or request.content_type == 'application/json':
            data = request.get_json(force=True, silent=True)
            if data and 'image' in data:
                image_data = data['image']
        elif request.data:
            data = request.get_json(force=True, silent=True)
            if data and 'image' in data:
                image_data = data['image']

        if not image_data:
            return jsonify({'error': 'Image manquante'}), 400

        emotion, scores, features, error = process_image(image_data)

        if error:
            return jsonify({
                'error': error,
                'action': 'MAINTENIR_TEMP',
                'comfort_action': 'MAINTENIR_TEMP'
            }), 200

        if emotion == 'angry':
            action = "AUGMENTER_TEMP"
        elif emotion == 'happy':
            action = "DIMINUER_TEMP"
        else:
            action = "MAINTENIR_TEMP"

        scores_percent = {k: float(round(v * 100, 1)) for k, v in scores.items()}
        features_clean = {k: float(round(v, 3)) for k, v in features.items()} if features else None

        return jsonify({
            'emotion': emotion,
            'scores': scores_percent,
            'action': action,
            'comfort_action': action,
            'features': features_clean,
            'method': 'opencv_haar'
        })

    except Exception as e:
        import traceback
        print("ERREUR:", str(e))
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'action': 'MAINTENIR_TEMP'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'method': 'opencv_haar'})


if __name__ == '__main__':
    print("=" * 50)
    print("API Détection d'Émotions - OpenCV Simple")
    print("=" * 50)
    print("Méthode: Haar Cascades + Analyse d'image")
    print("Émotions: angry, neutral, happy")
    print("Port: 5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
