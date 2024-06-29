import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import config
from preprocessing.data_loader import DataLoader, load_data
from models.model_builer import save_model, load_existing_model
from training.trainer import compile_model, train_model
from utils.functions import plot_history, make_predictions
import logging

logging.basicConfig(level=logging.INFO)

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(4, activation='relu')(inputs)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():
    logging.info("Loading data...")
    data_loader = DataLoader()
    X_train, Y_train = data_loader.load_data()

    input_shape = X_train.shape[1:]
    
    logging.info("Building the model...")
    model = build_model(input_shape=input_shape)
    
    compile_model(model)

    logging.info("Training the model...")
    history = train_model(model, X_train, Y_train, epochs=config.NUM_EPOCHS)
    plot_history(history)

    
    logging.info(f"Saving the model to {config.MODEL_PATH}...")
    save_model(model, config.MODEL_PATH)

    logging.info(f"Loading the model from {config.MODEL_PATH}...")
    loaded_model = load_existing_model(config.MODEL_PATH)
    
    
    threshold = 0.5
    
    raw_predictions, binary_predictions = make_predictions(loaded_model, X_train, threshold)

    logging.info("Raw Predictions:")
    print(raw_predictions)
    logging.info(f"Binary Predictions (threshold = {threshold}):")
    print(binary_predictions)

    logging.info("Comparison:")
    for x, raw, binary in zip(X_train, raw_predictions, binary_predictions):
        print(f"Input: {x} -> Raw: {raw[0]:.4f} -> Binary: {binary[0]}")

if __name__ == "__main__":
    main()

