# CodeClauseInternship_Handwritten_Digits_Recognition-0-9-

A Artificial Intelligence and machine learning project to recognize handwritten digits using a Convolutional Neural Network (CNN) model. The project uses an image as input and predicts the digit shown in the image.

## Project Overview

This project leverages a CNN model, which is well-suited for image classification tasks, specifically trained on the MNIST dataset containing thousands of labeled images of handwritten digits (0-9). Using this trained model, the program can recognize digits from any grayscale image input, predicting the correct number with high accuracy.

## Challenges Faced

1. **Model Training**: Ensuring the CNN architecture had adequate depth and complexity to handle the nuances of handwritten digits without overfitting on the MNIST dataset.
2. **Data Preprocessing**: Properly resizing, normalizing, and reshaping images from different sources to match the model’s input specifications (28x28 pixels) was essential for consistent prediction results.
3. **Image Quality Variations**: When tested with images outside the MNIST dataset, the model sometimes faced difficulty with noisy or poor-quality inputs, leading to lower prediction accuracy.
4. **Model Saving and Loading**: The project required careful handling to save and load models as `.h5` files to maintain consistency across different systems.

## Features

- **Digit Prediction**: Predicts digits from images of handwritten numbers.
- **CNN Model**: Uses a CNN trained on MNIST data for high accuracy.
- **Simple Interface**: Users only need to supply an image file path to make predictions.

## Getting Started

### Prerequisites

Ensure you have the following installed on your machine:
- Python 3.6 or above
- [TensorFlow](https://www.tensorflow.org/install) (latest version)
- [OpenCV](https://pypi.org/project/opencv-python/) (for image handling)
- [NumPy](https://pypi.org/project/numpy/) (for array manipulation)

To install required packages, run:

```bash
pip install tensorflow opencv-python numpy
```

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Handwritten_Digits_Recognition.git
   cd Handwritten_Digits_Recognition
   ```

2. **Download the Model File**:
   - Ensure that `mnist_digit_model.h5` (the trained model file) is in the root directory of the project. You can download a pre-trained model on MNIST if you haven’t trained your own model.

3. **Run the Program**:
   Use the following command to run the program, which will load an image and display the predicted digit.

   ```python
   python predict_digit.py --image_path "path/to/your/image.jpg"
   ```

### Running the Project

Follow these steps to predict a handwritten digit from an image:

1. Place your image (28x28 grayscale image of a single digit) in the project directory or specify the path to the image.
2. Run the `predict_and_display_digit` function from the script:
   
   ```python
   python predict_digit.py --image_path "path/to/your/image.jpg"
   ```

3. The program will load the model, preprocess the image, predict the digit, and display the image with the predicted digit overlaid on it.

### Project Files

- **mnist_digit_model.h5**: Pre-trained CNN model for digit recognition.
- **predict_digit.py**: Python script that loads the model, processes an image, and displays the predicted result.

## Example

Suppose you have an image at `Photos/3.jpg` with a handwritten "3". Run the following:

```bash
python predict_digit.py --image_path "Photos/3.jpg"
```

The program will display the image along with the predicted digit (e.g., "Predicted: 3").

## Future Improvements

- Extend the model to handle colored images and various image resolutions.
- Add GUI support for an easier user interface.
- Improve robustness for noisy or poorly scanned inputs.

## License

This project is open-source and available under the MIT License.

---

Replace `"your-username"` with your actual GitHub username and adjust paths as needed for your specific setup! Let me know if you need further modifications.
