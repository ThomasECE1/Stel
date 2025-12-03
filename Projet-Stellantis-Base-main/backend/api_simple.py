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
    Adapte les seuils en fonction de la luminosité
    """
    h, w = face_roi_gray.shape

    # Mesurer la luminosité moyenne du visage
    brightness = np.mean(face_roi_gray)

    # Adapter les seuils selon la luminosité
    # Plus c'est lumineux, plus les seuils doivent être élevés (moins sensible)
    if brightness > 130:  # Très lumineux
        brightness_factor = 2.0  # Seuils 100% plus hauts
    elif brightness > 100:  # Lumineux
        brightness_factor = 1.6  # Seuils 60% plus hauts
    elif brightness > 80:  # Normal
        brightness_factor = 1.2  # Seuils 20% plus hauts
    else:  # Sombre
        brightness_factor = 1.0  # Seuils normaux

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
        # Seuil adapté à la luminosité
        frown_indicators += min(vertical_lines / (30 * brightness_factor), 0.4)

    # 2. Contraste élevé dans la zone glabelle (rides = fort contraste)
    if glabella_region.size > 0:
        glabella_std = np.std(glabella_region)
        # Seuils adaptés à la luminosité
        if glabella_std > 35 * brightness_factor:
            frown_indicators += 0.3
        elif glabella_std > 25 * brightness_factor:
            frown_indicators += 0.2
        elif glabella_std > 15 * brightness_factor:
            frown_indicators += 0.1

    # 3. Assombrissement de la zone glabelle (ombre des sourcils froncés)
    if glabella_region.size > 0 and left_brow.size > 0 and right_brow.size > 0:
        glabella_mean = np.mean(glabella_region)
        brows_mean = (np.mean(left_brow) + np.mean(right_brow)) / 2
        # Seuils adaptés à la luminosité
        if glabella_mean < brows_mean - (10 * brightness_factor):
            frown_indicators += 0.3
        elif glabella_mean < brows_mean - (5 * brightness_factor):
            frown_indicators += 0.15

    # 4. Texture rugueuse dans la région des sourcils
    if glabella_region.size > 0:
        laplacian = cv2.Laplacian(glabella_region, cv2.CV_64F)
        texture_score = np.var(laplacian)
        # Seuils adaptés à la luminosité
        if texture_score > 500 * brightness_factor:
            frown_indicators += 0.2
        elif texture_score > 300 * brightness_factor:
            frown_indicators += 0.1

    return min(frown_indicators, 1.0), brightness


def analyze_mouth_shape(face_roi_gray):
    """
    Analyse la forme de la bouche
    Retourne: aspect_ratio, mouth_curve
    """
    h, w = face_roi_gray.shape
    mouth_region = face_roi_gray[int(h*0.6):int(h*0.9), int(w*0.2):int(w*0.8)]
    mh, mw = mouth_region.shape

    if mh == 0 or mw == 0:
        return 2.0, 0.0

    # Aspect ratio via contours
    blurred = cv2.GaussianBlur(mouth_region, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    aspect_ratio = 2.0
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        if ch > 0:
            aspect_ratio = cw / ch

    # === NOUVELLE MÉTHODE pour détecter la courbure ===
    # Utiliser le gradient vertical pour trouver la ligne des lèvres
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Diviser en 3 zones: coins gauche, centre, coins droit
    third = mw // 3
    left_zone = sobel_y[:, :third]
    center_zone = sobel_y[:, third:2*third]
    right_zone = sobel_y[:, 2*third:]

    def find_lip_line_y(zone):
        """Trouve la position Y de la ligne des lèvres via le gradient max"""
        if zone.size == 0:
            return zone.shape[0] / 2 if zone.shape[0] > 0 else 0
        # Somme horizontale des gradients pour chaque ligne
        row_gradients = np.sum(np.abs(zone), axis=1)
        if len(row_gradients) > 0 and np.max(row_gradients) > 0:
            # Trouver la ligne avec le gradient le plus fort (= ligne des lèvres)
            return np.argmax(row_gradients)
        return zone.shape[0] / 2

    left_y = find_lip_line_y(left_zone)
    center_y = find_lip_line_y(center_zone)
    right_y = find_lip_line_y(right_zone)

    # Courbure: si les coins sont plus HAUT (y plus petit) que le centre = sourire
    # (dans l'image, y=0 est en haut)
    avg_corners_y = (left_y + right_y) / 2
    mouth_curve = (center_y - avg_corners_y) / mh if mh > 0 else 0

    # Méthode alternative: analyser la forme générale
    # Chercher si les joues sont relevées (plis nasolabiaux)
    cheek_left = face_roi_gray[int(h*0.45):int(h*0.65), int(w*0.1):int(w*0.3)]
    cheek_right = face_roi_gray[int(h*0.45):int(h*0.65), int(w*0.7):int(w*0.9)]

    cheek_contrast = 0
    if cheek_left.size > 0 and cheek_right.size > 0:
        # Un sourire crée des plis = plus de contraste dans les joues
        cheek_contrast = (np.std(cheek_left) + np.std(cheek_right)) / 2

    # Bonus pour le sourire si fort contraste dans les joues
    if cheek_contrast > 35:
        mouth_curve += 0.03
    elif cheek_contrast > 25:
        mouth_curve += 0.02

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
    - SAD: coins de bouche vers le bas
    - ANGRY: sourcils froncés (critère principal)
    - NEUTRAL: expression neutre
    """
    scores = {'angry': 0.0, 'neutral': 0.0, 'happy': 0.0}

    # === HAPPY ===
    # Sourire détecté par Haar Cascade
    if smile_score > 0.3:
        scores['happy'] += 0.5
    elif smile_score > 0.15:
        scores['happy'] += 0.35
    elif smile_score > 0.05:
        scores['happy'] += 0.2

    # Coins de bouche relevés (critère principal pour sourire bouche fermée)
    if mouth_curve > 0.05:
        scores['happy'] += 0.55
    elif mouth_curve > 0.03:
        scores['happy'] += 0.45
    elif mouth_curve > 0.02:
        scores['happy'] += 0.3
    elif mouth_curve > 0.01:
        scores['happy'] += 0.15

    # Bouche large (sourire)
    if mouth_ratio > 3.5:
        scores['happy'] += 0.2
    elif mouth_ratio > 3.0:
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
        scores['neutral'] += 0.1

    # Sourcils détendus (pas froncés)
    if frown_score < 0.3:
        scores['neutral'] += 0.2
    elif frown_score < 0.38:
        scores['neutral'] += 0.1

    # Bouche droite (seuil plus strict)
    if -0.015 < mouth_curve < 0.015:
        scores['neutral'] += 0.15

    # Base neutre
    scores['neutral'] += 0.1

    # Normaliser
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    emotion = max(scores, key=scores.get)
    return emotion, scores


def draw_face_mesh(img, x, y, w, h):
    """
    Dessine un masque géométrique simplifié sur le visage détecté.
    Version légère pour éviter le lag.
    """
    color = (0, 255, 100)
    thickness = 1
    cx = x + w // 2

    # Seulement les points essentiels
    pts = [
        (cx, y + int(h * 0.02)),                    # top
        (x + int(w * 0.15), y + int(h * 0.15)),    # top_left
        (x + int(w * 0.85), y + int(h * 0.15)),    # top_right
        (x + int(w * 0.05), y + int(h * 0.4)),     # temple_left
        (x + int(w * 0.95), y + int(h * 0.4)),     # temple_right
        (x + int(w * 0.08), y + int(h * 0.65)),    # jaw_left
        (x + int(w * 0.92), y + int(h * 0.65)),    # jaw_right
        (x + int(w * 0.2), y + int(h * 0.85)),     # jaw_left_low
        (x + int(w * 0.8), y + int(h * 0.85)),     # jaw_right_low
        (cx, y + int(h * 0.98)),                    # chin
    ]

    # Contour du visage (gauche puis droite)
    contour_left = [pts[0], pts[1], pts[3], pts[5], pts[7], pts[9]]
    contour_right = [pts[0], pts[2], pts[4], pts[6], pts[8], pts[9]]

    for i in range(len(contour_left) - 1):
        cv2.line(img, contour_left[i], contour_left[i+1], color, thickness)
        cv2.line(img, contour_right[i], contour_right[i+1], color, thickness)

    # Ligne centrale verticale
    cv2.line(img, pts[0], pts[9], color, thickness)

    # Lignes horizontales (yeux, nez, bouche)
    cv2.line(img, pts[3], pts[4], color, thickness)  # niveau yeux
    cv2.line(img, pts[5], pts[6], color, thickness)  # niveau nez
    cv2.line(img, pts[7], pts[8], color, thickness)  # niveau bouche

    # Points aux intersections
    for pt in pts:
        cv2.circle(img, pt, 2, color, -1)

    return img


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
            return None, None, None, None, "Impossible de décoder l'image"

    except Exception as e:
        return None, None, None, None, f"Erreur: {str(e)}"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) == 0:
        return None, None, None, None, "Aucun visage détecté"

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y+h, x:x+w]

    smile_score = detect_smile(face_roi)
    frown_score, brightness = detect_frown(face_roi)
    mouth_ratio, mouth_curve = analyze_mouth_shape(face_roi)
    intensity = analyze_face_intensity(face_roi)

    emotion, scores = classify_emotion(smile_score, frown_score, mouth_ratio, intensity, mouth_curve)

    # Debug visuel avec barres de progression
    def make_bar(value, max_val=100, width=20):
        filled = int((value / max_val) * width)
        return '█' * filled + '░' * (width - filled)

    # Indicateur de luminosité
    if brightness > 150:
        light_icon = "☀️ TRÈS LUMINEUX"
    elif brightness > 120:
        light_icon = "🌤️ LUMINEUX"
    elif brightness > 90:
        light_icon = "⛅ NORMAL"
    else:
        light_icon = "🌙 SOMBRE"

    # Emojis pour chaque émotion
    emojis = {'angry': '😠', 'neutral': '😐', 'happy': '😊'}

    # Trouver l'émotion dominante
    winner = emojis.get(emotion, '❓')

    print(f"\n{'='*50}")
    print(f"  {winner} ÉMOTION DÉTECTÉE: {emotion.upper()} {winner}")
    print(f"{'='*50}")
    print(f"  💡 Luminosité: {brightness:.0f} ({light_icon})")
    print(f"  📊 Scores:")
    for emo in ['happy', 'neutral', 'angry']:
        pct = scores[emo] * 100
        bar = make_bar(pct)
        marker = " ◄" if emo == emotion else ""
        print(f"     {emojis[emo]} {emo.upper():8} {bar} {pct:5.1f}%{marker}")
    print(f"  ─────────────────────────────────────────")
    print(f"  📈 Métriques: smile={smile_score:.2f} | frown={frown_score:.2f} | curve={mouth_curve:+.3f}")
    print(f"{'='*50}\n")

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
