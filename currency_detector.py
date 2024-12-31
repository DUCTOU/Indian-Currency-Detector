import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from gtts import gTTS
import playsound

class IndianCurrencyClassifier:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.input_shape = (160, 160, 3)  # Reduced input size for faster processing
        self.model = None
        self.batch_size = 32

        train_dir = os.path.join(dataset_path, 'train')
        self.classes = sorted(os.listdir(train_dir))

    def prepare_data(self):
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            zoom_range=0.15,
            brightness_range=[0.8, 1.2]
        )

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'train'),
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        validation_generator = val_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'validation'),
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        return train_generator, validation_generator

    def create_model(self):
        # Using MobileNetV2 for feature extraction
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )

        # Freeze the base model
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(self.classes), activation='softmax')
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_model(self, epochs=20):
        train_generator, validation_generator = self.prepare_data()
        self.model = self.create_model()

        callbacks = [
            ModelCheckpoint(
                'best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                verbose=1
            )
        ]

        # Enable mixed precision training
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

        # Initial training
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Fine-tuning the model
        print("Fine-tuning the model...")
        base_model = self.model.layers[0]
        base_model.trainable = True

        # Freeze the first layers of the base model
        for layer in base_model.layers[:100]:
            layer.trainable = False

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        fine_tune_epochs = 10
        history_fine = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=fine_tune_epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Combine training history
        for key in history.history.keys():
            history.history[key].extend(history_fine.history[key])

        self.plot_training_history(history)
        return history

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

    def predict(self, image_path):
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=self.input_shape[:2]
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = self.model.predict(img_array, verbose=0)
        predicted_class = self.classes[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Text-to-Speech Integration
        announcement_text = f"This is a {predicted_class} rupee note with {confidence * 100:.2f} percent confidence."
        audio_file = "prediction_result.mp3"

        try:
            tts = gTTS(text=announcement_text, lang='en')
            tts.save(audio_file)
            playsound.playsound(audio_file)
        except Exception as e:
            print(f"Error during Text-to-Speech: {e}")
        finally:
            if os.path.exists(audio_file):
                os.remove(audio_file)  # Clean up the generated file

        return {
            "denomination": f"₹{predicted_class}",
            "confidence": confidence,
        }

if __name__ == "__main__":
    DATASET_PATH = r"E:\\WasteManagement\\Indian currency dataset v1"

    print("Initializing Currency Classifier...")
    classifier = IndianCurrencyClassifier(DATASET_PATH)

    print("Starting training process...")
    history = classifier.train_model()

    print("\nTesting model on sample image...")
    test_image_path = r"E:\\WasteManagement\\Indian currency dataset v1\\test\\20__9.jpg"
    result = classifier.predict(test_image_path)

    print(f"\nPrediction Results:")
    print(f"Predicted Denomination: {result['denomination']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
