# Dataset used: https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score




def create_cnn_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(9, 9, 1))
    x = tf.keras.layers.Conv2D(120, 2, activation='relu', padding="same")(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # 9x9 → 5x5

    x = tf.keras.layers.Conv2D(60, 3, activation='relu', padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)  # 5x5 → 3x3

    x = tf.keras.layers.Conv2D(30, 4, activation='relu', padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn')

    cnn_model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')

    return cnn_model


print(tf.config.list_physical_devices())

df = pd.read_csv('/home/lucas/ML-project/IntrusionDetection/datasets/cicids2017_cleaned_v2.csv')


label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

X = df.drop(columns=['Label'])
y = df['Label']


num_classes = len(np.unique(y))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pad_size = 81 - X_scaled.shape[1] 
X_padded = np.pad(X_scaled, ((0, 0), (0, pad_size)), mode='constant')
X_cnn = X_padded.reshape(-1, 9, 9, 1)


y_encoded = tf.keras.utils.to_categorical(y)
class_names = label_encoder.classes_

k = 5

kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


fold_accuracies = []
fold_precision = []
fold_recall = []
fold_f1 = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_cnn, y)):
    print(f"\nFold {fold + 1}/{k}")

    X_train, X_val = X_cnn[train_idx], X_cnn[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
    y_val_int = y[val_idx]

    model = create_cnn_model()

    tf.keras.utils.plot_model(model, to_file="cnn_with_maxpooling_architecture.png", show_shapes=True, show_layer_names=True)
    
    history = model.fit(X_train, y_train, epochs=30, batch_size=1024, validation_data=(X_val, y_val), verbose=1)

    score = model.evaluate(X_val, y_val, verbose=0)
    print(f"Fold {fold+1} Accuracy: {score[1] * 100:.3f}%")
    fold_accuracies.append(score[1])

    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true_int = y_val.argmax(axis=1)

    
    fold_precision.append(precision_score(y_true_int, y_pred, average='macro', zero_division=0))
    fold_recall.append(recall_score(y_true_int, y_pred, average='macro', zero_division=0))
    fold_f1.append(f1_score(y_true_int, y_pred, average='macro', zero_division=0))


    # Confusion Matrix
    cm = confusion_matrix(y_val_int, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Fold {fold+1} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    plt.savefig(f'confusion_matrix_fold_{fold+1}.png')
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy subplot
    ax1.plot(history.history['categorical_accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'Fold {fold+1} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss subplot
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'Fold {fold+1} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    plt.savefig(f'accloss_plot_fold_{fold+1}.png')
    plt.close()

    print(classification_report(y_val_int, y_pred))



print(f"\n Average Accuracy over {k} folds: {np.mean(fold_accuracies) * 100:.3f}%")
print(f"\n Average Precision over {k} folds: {np.mean(fold_precision) * 100:.3f}%")
print(f"\n Average Recall over {k} folds: {np.mean(fold_recall) * 100:.3f}%")
print(f"\n Average F1-Score over {k} folds: {np.mean(fold_f1) * 100:.3f}%")
