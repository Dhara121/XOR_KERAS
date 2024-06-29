import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy

def compile_model(model):
    model.compile(optimizer=SGD(learning_rate=0.1),
                  loss=BinaryCrossentropy(),
                  metrics=[BinaryAccuracy()])  # Adding accuracy metric

def train_model(model, X_train, Y_train, epochs=100):
    history = model.fit(X_train, Y_train, epochs=epochs, verbose=2)  # Increase verbosity for better logging
    return history

def evaluate_model(model, X_test, Y_test):
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=2)
    print(f'Test Accuracy: {accuracy:.4f}')
    return loss, accuracy
