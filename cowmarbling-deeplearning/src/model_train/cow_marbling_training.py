import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time

start_time = time.time()

base_dir = "C:\\Users\\kimyoungjin\\Desktop\\cow datasets\\"  # 데이터셋 폴더 경로 (수정 필요)

train_datagen = ImageDataGenerator(
    rescale=1.0/255,         # 픽셀 값을 [0,1]로 정규화
    rotation_range=20,       # 이미지 회전
    width_shift_range=0.2,   # 가로 이동
    height_shift_range=0.2,  # 세로 이동
    shear_range=0.2,         # 이미지 왜곡
    zoom_range=0.2,          # 확대/축소
    horizontal_flip=True,    # 좌우 반전
    validation_split=0.1     # 데이터의 10%를 검증 데이터로 사용
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

model.save("cow_marbling_resnet_model.h5")
print("Model saved as cow_marbling_resnet_model.h5")
end_time = time.time()

print("Time elapsed: {:.2f} seconds".format(end_time - start_time))
