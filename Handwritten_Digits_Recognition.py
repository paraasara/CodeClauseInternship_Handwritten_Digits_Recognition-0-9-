import numpy as np
import tensorflow as tf
import cv2  # OpenCV for image processing

# Define or load the CNN model (as in your code)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Function to predict the digit from an image and display it
def predict_and_display_digit(image_path):
    # Step 1: Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Verify that the image was loaded
    if img is None:
        print("Error: Image could not be loaded. Please check the file path.")
        return

    # Preprocess: Resize, normalize, and reshape
    img_resized = cv2.resize(img, (28, 28))
    img_normalized = img_resized / 255.0
    img_reshaped = img_normalized.reshape(1, 28, 28, 1)

    # Predict the digit
    prediction = model.predict(img_reshaped)
    digit = np.argmax(prediction)

    # Convert grayscale image to BGR for color text overlay
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Display the prediction on the image
    cv2.putText(img_bgr, f"Predicted: {digit}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 0), 2)  # Green text for high contrast

    # Show the image with the predicted digit
    cv2.imshow("Digit Prediction", img_bgr)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

# Example usage
predict_and_display_digit(r'C:\Users\ACER\Desktop\Suhas\Internship\Para internship\Photos\0.jpg')
