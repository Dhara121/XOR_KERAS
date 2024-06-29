import matplotlib.pyplot as plt

def plot_history(history):
    plt.plot(history.history['binary_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

def make_predictions(model, X,threshold=0.5):
    predictions = model.predict(X)
    binary_predictions = (predictions > threshold).astype(int)

    return predictions,binary_predictions
