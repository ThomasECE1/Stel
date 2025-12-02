import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import numpy as np
from PIL import Image
import io
import tempfile

# --- CONFIGURATION ---
app = Flask(__name__)
CORS(app)

# Liste des émotions (même ordre que l'ancien code pour compatibilité)
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print("✅ API DeepFace prête!")


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Aucun fichier 'image' trouvé"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    try:
        # Sauvegarder temporairement l'image (DeepFace a besoin d'un fichier ou array)
        image = Image.open(io.BytesIO(image_bytes))

        # Convertir en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Sauvegarder dans un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        # Analyse avec DeepFace
        result = DeepFace.analyze(
            img_path=temp_path,
            actions=['emotion'],
            enforce_detection=False,  # Ne pas échouer si pas de visage détecté
            silent=True
        )

        # Supprimer le fichier temporaire
        os.unlink(temp_path)

        # Extraire les résultats (DeepFace retourne une liste)
        if isinstance(result, list):
            result = result[0]

        emotions = result.get('emotion', {})
        dominant_emotion = result.get('dominant_emotion', 'neutral')

        # Afficher les scores
        print("\n=== SCORES DeepFace ===")
        for emotion, score in emotions.items():
            bar = "█" * int(score / 5)
            print(f"{emotion:10}: {score:5.1f}% {bar}")
        print("=" * 30)

        # Trouver la probabilité max
        max_probability = max(emotions.values()) / 100.0

        # Déterminer l'action
        if dominant_emotion in ['angry', 'fear', 'sad', 'disgust']:
            action = "AUGMENTER_TEMP"
            message = f"Détecté {dominant_emotion} ({max_probability * 100:.1f}%). Augmentation de la température."
        elif dominant_emotion in ['happy', 'surprise']:
            action = "DIMINUER_TEMP"
            message = f"Détecté {dominant_emotion} ({max_probability * 100:.1f}%). Diminution de la température."
        else:  # neutral
            action = "MAINTENIR_TEMP"
            message = f"Détecté {dominant_emotion} ({max_probability * 100:.1f}%). Confort stable."

        # Calculer top 3
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        top_3_emotions = [{'emotion': e, 'confidence': round(s/100, 4)} for e, s in sorted_emotions]

        return jsonify({
            "emotion": dominant_emotion,
            "confidence": max_probability,
            "comfort_action": action,
            "message": message,
            "top_3_emotions": top_3_emotions,
            "raw_scores": {k: v/100 for k, v in emotions.items()}
        })

    except Exception as e:
        print(f"Erreur DeepFace: {e}")
        return jsonify({
            "emotion": "neutral",
            "confidence": 0.5,
            "comfort_action": "MAINTENIR_TEMP",
            "message": f"Erreur de détection: {str(e)}",
            "top_3_emotions": [],
            "raw_scores": {}
        })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Démarrage API DeepFace sur le port 5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000)
