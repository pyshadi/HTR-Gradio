import gradio as gr
import tensorflow as tf
import numpy as np

# Load pre-trained MNIST model
model = tf.keras.models.load_model('../mnist_model.h5')


def classify_digit(img):
    # Ensure the image is 28x28 and invert colors
    img = np.array(img)
    img = np.resize(img, (28, 28))
    img = 255 - img
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0

    # Make predictions
    prediction = model.predict(img)
    predicted_label = int(np.argmax(prediction))  # Convert numpy.int64 to native int

    # Return the predicted label and the confidence scores for each class
    return predicted_label, {str(i): float(prediction[0][i]) for i in range(10)}

# Define Gradio Interface
iface = gr.Interface(
    fn=classify_digit,  # Function to be called on user input
    inputs=gr.Image(image_mode='L'),  # Input type and properties
    outputs=[gr.Label(), gr.JSON(label='Confidence Scores')],  # Output type and properties
    live=True  # Enable live mode
)
# Launch the interface
iface.launch()
