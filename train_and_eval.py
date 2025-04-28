import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from model_architecture import build_hybrid_model
import pickle


def load_datasets(binary):
    """
    Load train, validation, and test datasets from preprocessed data
    @params:
        binary: a boolean indicating if you want full data or just the 
    @returns:
        train_images, train_labels, val_images, val_labels, test_images, test_images: preprocessed data tensors
    """
    with open('../train_images.pkl', 'rb') as f:
            train_images = pickle.load(f)
    with open('../val_images.pkl', 'rb') as f:
            val_images = pickle.load(f)
    with open('../test_images.pkl', 'rb') as f:
            test_images = pickle.load(f)

    if binary:
        with open('../binary_train_labels.pkl', 'rb') as f:
            train_labels = pickle.load(f)
        train_labels = tf.cast(train_labels, dtype=tf.float32)

        with open('../binary_val_labels.pkl', 'rb') as f:
            val_labels = pickle.load(f)
        val_labels = tf.cast(val_labels, dtype=tf.float32)
            
        with open('../binary_test_labels.pkl', 'rb') as f:
            test_labels = pickle.load(f)
        test_labels = tf.cast(test_labels, dtype=tf.float32)

    else: 
        with open('../train_labels.pkl', 'rb') as f:
            train_labels = pickle.load(f)
        with open('../val_labels.pkl', 'rb') as f:
            val_labels = pickle.load(f)
        with open('../test_labels.pkl', 'rb') as f:
            test_labels = pickle.load(f)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def train_and_evaluate():
    binary = True # True for health/unhealthy, False for all 33 classes

    if binary:
        model_path = 'best_model_binary.keras'
        epochs = 15
    else:
        model_path = 'best_model_all_classes.keras'
        epochs = 50
    input_shape = (256, 256, 3)
    batch_size = 32

    # Load datasets
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_datasets(binary=binary)
    
    # Build model
    model = build_hybrid_model(input_shape=input_shape, binary=binary)

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=[
                    'accuracy',
                    tf.keras.metrics.F1Score(average='weighted')
                  ])

    # Callbacks
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', verbose=1)
    early_stop = EarlyStopping(patience=7, restore_best_weights=True, monitor='val_accuracy', verbose=1)

    # Train model
    history = model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        epochs=epochs,
        callbacks=[checkpoint, early_stop]
        )

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"training_plot_binary_{binary}.png")
    plt.show()

    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    best_model = tf.keras.models.load_model(model_path)
    test_loss, test_acc, test_f1 = best_model.evaluate(test_data, test_labels)
    print(f"✅ Test Accuracy: {test_acc:.4f}")
    print(f"📉 Test Loss: {test_loss:.4f}")
    print(f"Test Weighted F1 Score: {test_f1:.4f}")
    print(model.summary())

if __name__ == '__main__':
    train_and_evaluate()   