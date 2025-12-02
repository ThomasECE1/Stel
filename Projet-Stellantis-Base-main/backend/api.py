import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout


# =========================================================
# 1. DÉFINITION DE LA FONCTION PERSONNALISÉE
# =========================================================
def grayscale_to_rgb(x):
    return tf.concat([x, x, x], axis=-1)


# --- 2. CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_emotion_model.h5')
IMG_SIZE = (48, 48)

app = Flask(__name__)
CORS(app)

# Liste des classes d'émotions dans le bon ordre
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- 3. CHARGEMENT DU MODÈLE (UNE SEULE FOIS AU DÉMARRAGE) ---
try:
    MODEL = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'grayscale_to_rgb': grayscale_to_rgb}
    )
    print(f"✅ Modèle {MODEL_PATH} chargé avec succès.")
except Exception as e:
    print(f"🛑 ERREUR : Impossible de charger le modèle. {e}")
    MODEL = None


# --- 4. FONCTION DE PRÉTRAITEMENT D'IMAGE (CORRIGÉE) ---
def preprocess_image(image_bytes):
    """Prépare l'image reçue du Frontend pour le modèle CNN."""
    try:
        # 1. Lecture de l'image (PIL/Pillow)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize(IMG_SIZE).convert('L')  # Redimensionne et convertit en Grayscale (1 canal)

        # 2. Conversion en tableau Numpy (Forme attendue: (48, 48))
        image_array = np.array(image, dtype=np.float32)

        # 3. PAS DE NORMALISATION - TensorFlow image_dataset_from_directory
        # garde les valeurs 0-255, donc on garde les mêmes valeurs
        # (Ne pas diviser par 255 ici car le modèle a été entraîné avec des valeurs 0-255)

        # 4. Ajout des dimensions de canal (1) et de batch (1)
        image_array = np.expand_dims(image_array, axis=-1)  # Ajoute la dimension Canal (1) -> forme (48, 48, 1)
        image_array = np.expand_dims(image_array, axis=0)  # Ajoute la dimension Batch (1) -> forme (1, 48, 48, 1)

        return image_array
    except Exception as e:
        print(f"Erreur de prétraitement: {e}")
        return None


# --- 5. ROUTE API POUR LA PRÉDICTION ---
@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({"error": "Modèle non chargé"}), 503

    if 'image' not in request.files:
        return jsonify({"error": "Aucun fichier 'image' trouvé"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    processed_image = preprocess_image(image_bytes)

    if processed_image is None:
        return jsonify({"error": "Image non valide ou erreur de traitement"}), 400

    # Faire la prédiction
    predictions = MODEL.predict(processed_image, verbose=0)

    # ----------------------------------------------------
    # DEBUG : Afficher tous les scores
    # ----------------------------------------------------
    print("\n=== SCORES DE PRÉDICTION ===")
    for i, emotion in enumerate(EMOTION_CLASSES):
        score = predictions[0][i] * 100
        bar = "█" * int(score / 5)
        print(f"{emotion:10}: {score:5.1f}% {bar}")
    print("=" * 30)

    # Probabilité maximale et émotion prédite
    max_probability = np.max(predictions[0])
    predicted_class_index = np.argmax(predictions[0])
    predicted_emotion = EMOTION_CLASSES[predicted_class_index]

    # Calcul des top 3 émotions pour le débogage
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]  # Indice des 3 plus hauts scores
    top_3_emotions = [
        {'emotion': EMOTION_CLASSES[i], 'confidence': round(predictions[0][i].item(), 4)}
        for i in top_3_indices
    ]

    # LOGIQUE SIMPLE : Utiliser l'émotion avec le score le plus élevé
    final_emotion = predicted_emotion

    if predicted_emotion in ['angry', 'fear', 'sad', 'disgust']:
        # Émotions négatives → augmenter la température
        action = "AUGMENTER_TEMP"
        message = f"Détecté {predicted_emotion} ({max_probability * 100:.1f}%). Augmentation de la température."
    elif predicted_emotion in ['happy', 'surprise']:
        # Émotions positives → diminuer la température
        action = "DIMINUER_TEMP"
        message = f"Détecté {predicted_emotion} ({max_probability * 100:.1f}%). Diminution de la température."
    else:  # neutral
        action = "MAINTENIR_TEMP"
        message = f"Détecté {predicted_emotion} ({max_probability * 100:.1f}%). Confort stable."

    return jsonify({
        "emotion": final_emotion,
        "confidence": max_probability.tolist(),
        "comfort_action": action,
        "message": message,
        "top_3_emotions": top_3_emotions,  # Nouvelle clé pour le débogage
        "raw_scores": predictions[0].tolist()
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)