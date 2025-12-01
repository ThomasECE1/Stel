import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Lambda
from tensorflow.keras.applications import ResNet50
import os

# --- 1. CONFIGURATION INTELLIGENTE ---
# Trouver le chemin d'accès au dossier de base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

print(f"Le script cherche les données ici : {DATA_DIR}")

# Paramètres du Dataset
BATCH_SIZE = 64
IMG_SIZE = (48, 48)
NUM_CLASSES = 7


def load_data():
    """Charge les données directement depuis les dossiers."""
    print("Chargement des données depuis les dossiers...")

    # Création du générateur d'images pour l'ENTRAÎNEMENT
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'train'),
        labels='inferred',
        label_mode='categorical',
        image_size=IMG_SIZE,
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        color_mode='grayscale'
    )

    # Création du générateur d'images pour le TEST / VALIDATION
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'test'),
        labels='inferred',
        label_mode='categorical',
        image_size=IMG_SIZE,
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        color_mode='grayscale'
    )

    print(f"Classes détectées: {train_ds.class_names}")
    return train_ds, val_ds


def build_fine_tuned_model():
    """Construit le modèle CNN ResNet-50 pour le Transfer Learning."""
    print("Construction du modèle (Transfer Learning)...")

    # --- 1. ADAPTATION POUR LE GRIS (1 canal -> 3 canaux) ---
    input_tensor = Input(shape=(48, 48, 1))

    # Correction de l'erreur : Utilisation de Lambda pour wrapper tf.concat
    def grayscale_to_rgb(x):
        # Duplication du canal gris pour simuler 3 canaux (RGB)
        return tf.concat([x, x, x], axis=-1)

    x = Lambda(grayscale_to_rgb)(input_tensor)

    # --- 2. TÉLÉCHARGEMENT DU BACKBONE (ResNet-50) ---
    base_model = ResNet50(weights='imagenet',  # Poids pré-entraînés pour le TL
                          include_top=False,  # On retire la tête de classification standard
                          input_tensor=x,  # On utilise notre entrée adaptée à 3 canaux
                          pooling='avg')

    # On gèle les poids du modèle pré-entraîné
    for layer in base_model.layers:
        layer.trainable = False

        # --- 3. AJOUTER LA NOUVELLE TÊTE (Fine-Tuning Head) ---
    x = base_model.output
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)  # 7 neurones de sortie pour les 7 classes

    model = Model(inputs=input_tensor, outputs=predictions)

    # --- 4. COMPILATION DU MODÈLE ---
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Modèle construit et compilé. Prêt pour l'entraînement.")
    return model


if __name__ == '__main__':
    # ----------------------------------------------------
    # PHASE D'EXÉCUTION PRINCIPALE
    # ----------------------------------------------------

    train_ds, val_ds = load_data()
    model = build_fine_tuned_model()

    # --- 5. ENTRAÎNEMENT DU MODÈLE (Fine-Tuning) ---
    print("\nLancement de l'entraînement (Fine-Tuning) sur les 7 classes...")

    # L'entraînement prendra du temps (minutes/heures selon le PC).
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10  # Nombre d'époques pour l'apprentissage
    )

    # --- 6. SAUVEGARDE DU MODÈLE ---
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'emotion_model_resnet.h5')
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Modèle sauvegardé avec succès dans : {MODEL_SAVE_PATH}")