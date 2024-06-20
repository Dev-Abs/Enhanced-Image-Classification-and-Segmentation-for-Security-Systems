import os
import cv2
import numpy as np
from skimage.feature import hog
from scipy.fftpack import fft2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
import joblib
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (128, 128)  # Resize images to this size for classification
IMAGE_HEIGHT = 256  # Image height for segmentation
IMAGE_WIDTH = 256  # Image width for segmentation

# Load the SVM model
with open('svm_model_en.pkl', 'rb') as f:
    svm_model = joblib.load(f)

# Load the U-Net model
unet_model = load_model('unet_model.h5')

# Extract HOG features
def extract_hog_features(images):
    hog_features = []
    for img in images:
        features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True, multichannel=False)
        hog_features.append(features)
    return np.array(hog_features)

# Extract Fourier Transform features
def extract_fourier_features(images):
    fourier_features = []
    for img in images:
        f_transform = np.abs(fft2(img))
        f_transform = f_transform.flatten()
        fourier_features.append(f_transform)
    return np.array(fourier_features)

# Extract CNN features using a pre-trained VGG16 model
def extract_cnn_features(images):
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model = Model(inputs=vgg16.input, outputs=vgg16.layers[-1].output)
    cnn_features = []
    for img in images:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_color = cv2.resize(img_color, IMAGE_SIZE)
        img_array = img_to_array(img_color)
        img_array = np.expand_dims(img_array, axis=0)
        features = model.predict(img_array)
        cnn_features.append(features.flatten())
    return np.array(cnn_features)

# Preprocess image for classification
def preprocess_image_classification(image):
    img = cv2.resize(image, IMAGE_SIZE)
    return img

# Classify and segment the image
def classify_and_segment(image):
    # Preprocess image for classification
    processed_image = preprocess_image_classification(image)
    processed_image = processed_image.reshape(1, *processed_image.shape)

    # Extract features
    hog_features = extract_hog_features(processed_image)
    fourier_features = extract_fourier_features(processed_image)
    cnn_features = extract_cnn_features(processed_image)
    combined_features = np.hstack((hog_features, fourier_features, cnn_features))

    # Classify image
    classification = svm_model.predict(combined_features)[0]

    if classification == 0:  # Safe
        return "safe", None, None

    # If classified as unsafe, perform segmentation
    processed_image_segmentation = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    processed_image_segmentation = np.expand_dims(processed_image_segmentation, axis=-1)
    processed_image_segmentation = np.expand_dims(processed_image_segmentation, axis=0)
    segmentation_result = unet_model.predict(processed_image_segmentation)
    segmented_class = np.argmax(segmentation_result, axis=-1).reshape((IMAGE_HEIGHT, IMAGE_WIDTH))

    if np.sum(segmented_class == 1) > np.sum(segmented_class == 2) and np.sum(segmented_class == 1) > np.sum(segmented_class == 3):
        return "unsafe", "GUN", segmented_class
    elif np.sum(segmented_class == 2) > np.sum(segmented_class == 1) and np.sum(segmented_class == 2) > np.sum(segmented_class == 3):
        return "unsafe", "knife", segmented_class
    elif np.sum(segmented_class == 3) > np.sum(segmented_class == 1) and np.sum(segmented_class == 3) > np.sum(segmented_class == 2):
        return "unsafe", "shuriken", segmented_class
    else:
        return "unsafe", "unknown", segmented_class

# Path to test data
test_folder = '/content/drive/MyDrive/dataset/test'

# Iterate through test images
for category in ['safe', 'GUN', 'knife', 'shuriken']:
    category_folder = os.path.join(test_folder, category)
    predicted_masks_folder = os.path.join(category_folder, 'predicted_masks')
    os.makedirs(predicted_masks_folder, exist_ok=True)

    for filename in os.listdir(category_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(category_folder, filename)
            test_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if test_image is None:
                continue

            result, object_type, segmented_mask = classify_and_segment(test_image)

            # Save and plot segmented image if classified as unsafe
            if result == "unsafe" and segmented_mask is not None:
                mask_path = os.path.join(predicted_masks_folder, filename)
                plt.imsave(mask_path, segmented_mask, cmap='gray')

                # Plot original and segmented images
                # plt.figure(figsize=(10, 5))
                # plt.subplot(1, 2, 1)
                # plt.title('Original Image')
                # plt.imshow(test_image, cmap='gray')
                # plt.subplot(1, 2, 2)
                # plt.title('Segmented Image')
                # plt.imshow(segmented_mask, cmap='gray')
                # plt.show()

            print(f"File: {file_path}, Result: {result}, Object Type: {object_type}")
