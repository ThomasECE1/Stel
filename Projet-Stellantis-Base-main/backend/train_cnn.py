import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

# --- 1. CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

print(f"Le script cherche les données ici : {DATA_DIR}")

# Paramètres du Dataset
BATCH_SIZE = 32  # Réduit pour meilleure généralisation
IMG_SIZE = (48, 48)
NUM_CLASSES = 7


# --- 2. DATA AUGMENTATION ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="data_augmentation")


def load_data():
    """Charge les données directement depuis les dossiers."""
    print("Chargement des données depuis les dossiers...")

    # Création du générateur d'images pour l'ENTRAÎNEMENT
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'train2', 'train'),
        labels='inferred',
        label_mode='categorical',
        image_size=IMG_SIZE,
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        shuffle=True,
        seed=42
    )

    # Création du générateur d'images pour le TEST / VALIDATION
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'test2', 'test'),
        labels='inferred',
        label_mode='categorical',
        image_size=IMG_SIZE,
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        color_mode='grayscale'
    )

    print(f"Classes détectées: {train_ds.class_names}")

    # Appliquer data augmentation sur le train set
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Optimiser les performances avec prefetch
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds


def build_fine_tuned_model():
    """Construit le modèle CNN ResNet-50 pour le Transfer Learning."""
    print("Construction du modèle (Transfer Learning)...")

    # --- 1. ADAPTATION POUR LE GRIS (1 canal -> 3 canaux) ---
    input_tensor = Input(shape=(48, 48, 1))

    def grayscale_to_rgb(x):
        return tf.concat([x, x, x], axis=-1)

    x = Lambda(grayscale_to_rgb)(input_tensor)

    # --- 2. TÉLÉCHARGEMENT DU BACKBONE (ResNet-50) ---
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=x,
        pooling='avg'
    )

    # Dégeler les dernières couches pour fine-tuning
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    # --- 3. NOUVELLE TÊTE AMÉLIORÉE ---
    x = base_model.output
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=predictions)

    # --- 4. COMPILATION AVEC LEARNING RATE OPTIMISÉ ---
    optimizer = Adam(learning_rate=0.0001)  # Learning rate plus bas pour fine-tuning

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Modèle construit et compilé. Prêt pour l'entraînement.")
    print(f"Couches entraînables: {sum([1 for l in model.layers if l.trainable])}")
    return model


if __name__ == '__main__':
    # ----------------------------------------------------
    # PHASE D'EXÉCUTION PRINCIPALE
    # ----------------------------------------------------

    train_ds, val_ds = load_data()
    model = build_fine_tuned_model()

    # --- 5. CALLBACKS POUR OPTIMISER L'ENTRAÎNEMENT ---
    callbacks = [
        # Arrêt anticipé si pas d'amélioration
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Réduire le learning rate si plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        # Sauvegarder le meilleur modèle
        ModelCheckpoint(
            os.path.join(BASE_DIR, 'best_emotion_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # --- 6. ENTRAÎNEMENT DU MODÈLE ---
    print("\nLancement de l'entraînement amélioré...")
    print("- Data Augmentation: activée")
    print("- Early Stopping: patience=5")
    print("- Learning Rate Scheduler: activé")
    print("- Epochs max: 30\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks
    )

    # --- 7. SAUVEGARDE DU MODÈLE FINAL ---
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'emotion_model_resnet.h5')
    model.save(MODEL_SAVE_PATH)

    # Afficher les résultats finaux
    print(f"\n{'='*50}")
    print(f"✅ Entraînement terminé!")
    print(f"✅ Meilleure accuracy validation: {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"✅ Modèle sauvegardé: {MODEL_SAVE_PATH}")
    print(f"✅ Meilleur modèle: {os.path.join(BASE_DIR, 'best_emotion_model.h5')}")
    print(f"{'='*50}")
