from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model


def build_model(input_shape):
    
    inputs = Input(shape=input_shape)
    hidden = Dense(8, activation='relu')(inputs)
    outputs = Dense(1, activation='sigmoid')(hidden)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def save_model(model, path):
    
    model.save(path)

def load_existing_model(path):
    
    return load_model(path)    