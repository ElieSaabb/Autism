# model/cnn3d.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, BatchNormalization,
    GlobalAveragePooling3D, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam

def build_3dcnn(input_shape=(64, 64, 64, 1), compile_model=True, lr=1e-3):
    inputs = Input(shape=input_shape)

    # Conv Block 1
    x = Conv3D(64, 3, padding="same", activation="relu")(inputs)
    x = MaxPooling3D(2)(x)
    x = BatchNormalization()(x)

    # Conv Block 2
    x = Conv3D(64, 3, padding="same", activation="relu")(x)
    x = MaxPooling3D(2)(x)
    x = BatchNormalization()(x)

    # Conv Block 3
    x = Conv3D(128, 3, padding="same", activation="relu")(x)
    x = MaxPooling3D(2)(x)
    x = BatchNormalization()(x)

    # Conv Block 4
    x = Conv3D(256, 3, padding="same", activation="relu")(x)
    x = MaxPooling3D(2)(x)
    x = BatchNormalization()(x)

    # Classification head
    x = GlobalAveragePooling3D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="ASD_3D_CNN")

    if compile_model:
        model.compile(optimizer=Adam(learning_rate=lr),
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
    
    return model