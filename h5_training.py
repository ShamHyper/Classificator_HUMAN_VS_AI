import tensorflow as tf

ver = "0.1"

print(f"H5-Trainer v{ver}|Starting training...")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

from tensorflow.keras.applications import VGG16  # type: ignore  # noqa: E402
from tensorflow.keras.models import Sequential  # type: ignore  # noqa: E402
from tensorflow.keras.layers import Dense, Flatten  # type: ignore # noqa: E402
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore # noqa: E402

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

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
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'data/train',  
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    'data/test', 
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  
)

model.fit(
    train_generator, 
    validation_data=validation_generator, 
    epochs=10  # changme
)

model.save('model_multiai.h5')

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")