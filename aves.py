# -*- coding: utf-8 -*-
"""Entrenamiento con MobileNetV2 para clasificación de aves."""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Rutas a los datos
TRAIN_DIRECTORY = "/LUSTRE/home/rn_lcc_11/share/birds/train"
TEST_DIRECTORY = "/LUSTRE/home/rn_lcc_11/share/birds/test"

# ---------------------------------------------------------------------
# 1. Cargar rutas de imágenes de entrenamiento y sus etiquetas
# ---------------------------------------------------------------------
datos = []

for nombre_ave in os.listdir(TRAIN_DIRECTORY):
    ruta_ave = os.path.join(TRAIN_DIRECTORY, nombre_ave)
    if os.path.isdir(ruta_ave):
        for archivo in os.listdir(ruta_ave):
            ruta_imagen = os.path.join(ruta_ave, archivo)
            if os.path.isfile(ruta_imagen):
                datos.append({
                    "ruta": ruta_imagen,
                    "etiqueta": nombre_ave
                })

df = pd.DataFrame(datos)
print(df.head())

# ---------------------------------------------------------------------
# 2. Separar en conjuntos de entrenamiento y validación
# ---------------------------------------------------------------------
df_train, df_val = train_test_split(df, test_size=0.2, stratify=df['etiqueta'], random_state=42)

# ---------------------------------------------------------------------
# 3. Crear generadores de datos con aumentos y preprocesamiento
# ---------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.8, 1.2],
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col='ruta',
    y_col='etiqueta',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=df_val,
    x_col='ruta',
    y_col='etiqueta',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ---------------------------------------------------------------------
# 4. Cargar MobileNetV2 base preentrenada sin la capa final
# ---------------------------------------------------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congelamos la base al principio

num_clases = len(train_generator.class_indices)

# Añadir nuevas capas encima de la base preentrenada
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_clases)(x)  # Sin softmax porque usamos from_logits=True

model = Model(inputs=base_model.input, outputs=outputs)

# Compilamos el modelo con un learning rate alto al inicio
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# ---------------------------------------------------------------------
# 5. Entrenamiento inicial (solo las capas nuevas)
# ---------------------------------------------------------------------
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3, verbose=1),
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks
)

# Guardamos gráfica de precisión y pérdida
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión (Accuracy)')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.savefig("Accuracy_y_Loss1.png")
plt.close()

# ---------------------------------------------------------------------
# 6. Fine-tuning: entrenar solo las últimas 10 capas del modelo base
# ---------------------------------------------------------------------
# Descongelamos solo las últimas 10 capas de MobileNetV2
for layer in base_model.layers[:-10]:
    layer.trainable = False

for layer in base_model.layers[-10:]:
    layer.trainable = True

# Recompilamos con un learning rate más bajo
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Entrenamos nuevamente
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks
)

# Guardamos gráfica de fine-tuning
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_finetune.history['accuracy'], label='Entrenamiento')
plt.plot(history_finetune.history['val_accuracy'], label='Validación')
plt.title('Precisión (Accuracy)')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_finetune.history['loss'], label='Entrenamiento')
plt.plot(history_finetune.history['val_loss'], label='Validación')
plt.title('Pérdida (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.savefig("Accuracy_y_Loss2.png")
plt.close()

# ---------------------------------------------------------------------
# 7. Preparar datos de prueba
# ---------------------------------------------------------------------
test = []

for nombre_ave in os.listdir(TEST_DIRECTORY):
    ruta_ave = os.path.join(TEST_DIRECTORY, nombre_ave)
    if os.path.isdir(ruta_ave):
        for archivo in os.listdir(ruta_ave):
            ruta_imagen = os.path.join(ruta_ave, archivo)
            if os.path.isfile(ruta_imagen):
                test.append({
                    "ruta": ruta_imagen,
                    "etiqueta": nombre_ave
                })

df1 = pd.DataFrame(test)
print(df1.head())

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df1,
    x_col='ruta',
    y_col='etiqueta',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ---------------------------------------------------------------------
# 8. Evaluación del modelo sobre el conjunto de prueba
# ---------------------------------------------------------------------
preds = model.predict(test_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=class_labels))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues")
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.savefig("Confusion.png")
plt.close()

# ---------------------------------------------------------------------
# 9. Guardar el modelo entrenado
# ---------------------------------------------------------------------
model.save("modelo.h5")
