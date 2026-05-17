import tensorflow as tf
from tensorflow.keras import layers

def get_data_augmentation(input_shape=(32, 32, 3)):
    data_augmentation = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ])
    return data_augmentation

def create_augmented_model_cifar10(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    
    x = get_data_augmentation(input_shape)(inputs)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def show_augmentation_examples(x_train, num_examples=5):
    import matplotlib.pyplot as plt
    import numpy as np
    
    aug = get_data_augmentation()
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(9, num_examples * 3))
    
    for i in range(num_examples):
        idx = np.random.randint(0, len(x_train))
        original = x_train[idx:idx+1]
        
        axes[i, 0].imshow(original[0])
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        for j in range(1, 3):
            augmented = aug(original, training=True)
            axes[i, j].imshow(augmented[0])
            axes[i, j].set_title(f'Augmented {j}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('models/data_augmentation_examples.png')
    plt.close()
    print("Augmentation examples saved to models/data_augmentation_examples.png")

if __name__ == "__main__":
    from data.load_datasets import load_cifar10_data
    
    (x_train, y_train), (x_test, y_test), classes = load_cifar10_data()
    show_augmentation_examples(x_train)
