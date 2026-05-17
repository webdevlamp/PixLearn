import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from data.load_datasets import load_cifar10_data
from src.models import create_transfer_learning_model

def train_transfer_learning(base_model_name='MobileNetV2', epochs=15, batch_size=32):
    print(f"=== Training with Transfer Learning ({base_model_name}) ===")
    (x_train, y_train), (x_test, y_test), classes = load_cifar10_data()
    
    model = create_transfer_learning_model(
        input_shape=(32, 32, 3),
        num_classes=10,
        base_model_name=base_model_name
    )
    
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=2),
        ModelCheckpoint(f'models/{base_model_name}_best.h5', save_best_only=True)
    ]
    
    print(f"\nTraining {base_model_name} (frozen base)...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks
    )
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n{base_model_name} Test Accuracy (frozen): {test_acc:.4f}")
    
    model.save(f'models/{base_model_name}_frozen.h5')
    
    print(f"\nFine-tuning {base_model_name}...")
    base_model = model.layers[0]
    base_model.trainable = True
    
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_fine = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks
    )
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n{base_model_name} Test Accuracy (fine-tuned): {test_acc:.4f}")
    
    model.save(f'models/{base_model_name}_fine_tuned.h5')
    
    plot_transfer_learning_history(history, history_fine, base_model_name)
    
    return model

def plot_transfer_learning_history(history, history_fine, model_name):
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(acc, label='Train Accuracy')
    axes[0].plot(val_acc, label='Val Accuracy')
    axes[0].axvline(len(history.history['accuracy']), color='r', linestyle='--', label='Fine-tuning start')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'{model_name} Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(loss, label='Train Loss')
    axes[1].plot(val_loss, label='Val Loss')
    axes[1].axvline(len(history.history['loss']), color='r', linestyle='--', label='Fine-tuning start')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'{model_name} Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'models/{model_name}_transfer_learning_history.png')
    plt.close()
    print(f"Training history saved to models/{model_name}_transfer_learning_history.png")

if __name__ == "__main__":
    import tensorflow as tf
    os.makedirs('models', exist_ok=True)
    train_transfer_learning('MobileNetV2')
