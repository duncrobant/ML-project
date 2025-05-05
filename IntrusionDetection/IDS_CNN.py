# Dataset used: https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

from imblearn.under_sampling import RandomUnderSampler


def create_cnn_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(8, 8, 1))
    x = tf.keras.layers.Conv2D(120, 2, activation='relu', padding="same")(inputs)
    x = tf.keras.layers.Conv2D(60, 3, activation='relu', padding="same")(x)
    x = tf.keras.layers.Conv2D(30, 4, activation='relu', padding="same")(x)

    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(7, activation='softmax')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn')

    cnn_model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')

    return cnn_model


print(tf.__version__)
print(tf.config.list_physical_devices())

df = pd.read_csv('/home/lucas/ML-project/IntrusionDetection/datasets/cicids2017_cleaned.csv')

#df = df.sample(frac=1)

label_encoder = LabelEncoder()
df['Attack Type'] = label_encoder.fit_transform(df['Attack Type'])  # integer encoding


X = df.drop(columns=['Attack Type'])
y = df['Attack Type']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pad_size = 64 - 52  # => 12 zeros to pad
X_padded = np.pad(X_scaled, ((0, 0), (0, pad_size)), 'constant')
X_cnn = X_padded.reshape(-1, 8, 8, 1)

#X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)


y_encoded = tf.keras.utils.to_categorical(y)
class_names = label_encoder.classes_

k = 3

kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_cnn, y)):
    print(f"\nFold {fold + 1}/{k}")

    X_train, X_val = X_cnn[train_idx], X_cnn[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
    y_val_int = y[val_idx]

    model = create_cnn_model()
    history = model.fit(X_train, y_train, epochs=30, batch_size=1024, validation_data=(X_val, y_val), verbose=1)

    score = model.evaluate(X_val, y_val, verbose=0)
    print(f"Fold {fold+1} Accuracy: {score[1] * 100:.3f}%")
    fold_accuracies.append(score[1])

    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

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



#X_train, X_test, y_train, y_test = train_test_split(X_cnn, y_encoded, test_size=0.2, random_state=42, stratify=y)


print(f"\n Average Accuracy over {k} folds: {np.mean(fold_accuracies) * 100:.3f}%")

""" model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')  # for multi-class classification
]) 

print(f"\n📊 Average Accuracy over {k} folds: {np.mean(fold_accuracies) * 100:.2f}%")



#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()



loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.4f}%")

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

labels = ["Bots", "Brute Force", "DDoS", "DoS", "Normal Traffic", "Port Scanning", "Web Attacks"]

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('CNN Confusion Matrix')
plt.show()"""

