import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore

def predict(image_path):
    """
        Predicts the class of an image using the provided model.

        Args:
        model: A trained machine learning model for image classification.
        image_path: Path to the image file to be classified.

        Returns:
        str: The predicted class name of the image.

        Raises:
        None
    """
    model = tf.keras.models.load_model('../model/model.keras')  #
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Return the predicted class name
    return classes[predicted_class]

