import tensorflow as tf
from tensorflow.keras.applications import VGG16  # type: ignore  
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Flatten  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

ver = "0.4"
epochs = 10
datasets_sizes = (512, 512)
input_shape_sizes = (512, 512, 3)

print(f"H5-Trainer v{ver} | Starting training...")
print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(input_shape_sizes))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1.0/255, 
    validation_split=0.2 
)

train_generator = train_datagen.flow_from_directory(
    'data/train', 
    target_size=datasets_sizes,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'data/train',  
    target_size=datasets_sizes,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model.fit(
    train_generator, 
    validation_data=validation_generator, 
    epochs=epochs
)

model.save('model.h5')