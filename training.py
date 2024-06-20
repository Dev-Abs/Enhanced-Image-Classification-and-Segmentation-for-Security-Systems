import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.fftpack import fft2, ifft2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.utils import to_categorical

# Constants
IMAGE_SIZE = (128, 128)  # Resize images to this size for classification
IMAGE_HEIGHT = 256  # Image height for segmentation
IMAGE_WIDTH = 256  # Image width for segmentation
NUM_CLASSES = 4  # Safe (0), Gun (1), Knife (2), Shuriken (3)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# Load images and labels
def load_images_and_labels(folder):
    images = []
    labels = []
    label_mapping = {'safe': 0, 'unsafe': 1}
    unsafe_subcategories = ['knife', 'GUN', 'shuriken']

    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            if label == 'safe':
                for filename in os.listdir(label_folder):
                    img_path = os.path.join(label_folder, filename)
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, IMAGE_SIZE)
                            images.append(img)
                            labels.append(label_mapping[label])
            elif label == 'unsafe':
                for subcategory in unsafe_subcategories:
                    subcategory_folder = os.path.join(label_folder, subcategory)
                    if os.path.isdir(subcategory_folder):
                        for filename in os.listdir(subcategory_folder):
                            img_path = os.path.join(subcategory_folder, filename)
                            if filename.endswith(".jpg") or filename.endswith(".png"):
                                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                if img is not None:
                                    img = cv2.resize(img, IMAGE_SIZE)
                                    images.append(img)
                                    labels.append(label_mapping[label])

    # Balance the dataset
    images, labels = np.array(images), np.array(labels)
    safe_images = images[labels == 0]
    unsafe_images = images[labels == 1]

    if len(safe_images) > len(unsafe_images):
        safe_images = resample(safe_images, replace=False, n_samples=len(unsafe_images), random_state=42)
    else:
        unsafe_images = resample(unsafe_images, replace=True, n_samples=len(safe_images), random_state=42)

    balanced_images = np.concatenate((safe_images, unsafe_images))
    balanced_labels = np.array([0] * len(safe_images) + [1] * len(unsafe_images))

    return balanced_images, balanced_labels

# Extract HOG features
def extract_hog_features(images):
    hog_features = []
    for img in images:
        features, _ = hog(img, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_PIXELS_PER_CELL,
                          cells_per_block=HOG_CELLS_PER_BLOCK, visualize=True, multichannel=False)
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

# Load data
train_folder = '/content/drive/MyDrive/dataset/train'
images, labels = load_images_and_labels(train_folder)

# Extract features
hog_features = extract_hog_features(images)
fourier_features = extract_fourier_features(images)
cnn_features = extract_cnn_features(images)

# Combine all features
combined_features = np.hstack((hog_features, fourier_features, cnn_features))

# Split data
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)

# Train SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy}")

# Save the model
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

# Define U-Net model for segmentation
def unet_model(input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)):
    inputs = Input(input_size)

    # Encoder path
    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    # Decoder path
    u6 = UpSampling2D(size=(2, 2))(c3)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = UpSampling2D(size=(2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)

    outputs = Conv2D(NUM_CLASSES, 1, activation='softmax')(c7)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_images_and_masks(train_folder):
    images = []
    masks = []

    annotations_folder = os.path.join(train_folder, 'annotations')

    for class_folder in os.listdir(train_folder):
        if class_folder in ['knife', 'GUN', 'shuriken', 'safe']:
            class_images_folder = os.path.join(train_folder, class_folder)
            class_annotations_folder = os.path.join(annotations_folder, class_folder)

            for filename in os.listdir(class_images_folder):
                if filename.endswith(".png") and not filename.startswith('.'):
                    img_path = os.path.join(class_images_folder, filename)
                    mask_filename = os.path.splitext(filename)[0] + '.png'
                    mask_path = os.path.join(class_annotations_folder, mask_filename)

                    if os.path.isfile(mask_path):
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                        if img is not None and mask is not None:
                            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                            mask = cv2.resize(mask, (IMAGE_WIDTH, IMAGE_HEIGHT))

                            mask = np.expand_dims(mask, axis=-1)
                            images.append(img)
                            masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)
    return images, masks

def train_unet_model(train_folder, validation_split=0.2):
    images, masks = load_images_and_masks(train_folder)
    masks = masks.astype(np.uint8)

    masks = np.squeeze(masks)
    masks_one_hot = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], NUM_CLASSES))

    for c in range(NUM_CLASSES):
        masks_one_hot[masks == c, c] = 1

    masks = masks_one_hot

    x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=validation_split, random_state=42)

    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    model = unet_model(input_shape)
    model.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_val, y_val))

    model.save('unet_model.h5')

# Train the U-Net model
train_unet_model(train_folder)

# Load the SVM model
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Load the U-Net model
unet_model = load_model('unet_model.h5')

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
        return "safe", None

    # If classified as unsafe, perform segmentation
    processed_image_segmentation = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    processed_image_segmentation = np.expand_dims(processed_image_segmentation, axis=-1)
    processed_image_segmentation = np.expand_dims(processed_image_segmentation, axis=0)
    segmentation_result = unet_model.predict(processed_image_segmentation)

    # Determine the specific object
    segmented_class = np.argmax(segmentation_result, axis=-1).reshape((IMAGE_HEIGHT, IMAGE_WIDTH))

    if np.sum(segmented_class == 1) > np.sum(segmented_class == 2) and np.sum(segmented_class == 1) > np.sum(segmented_class == 3):
        return "unsafe", "GUN"
    elif np.sum(segmented_class == 2) > np.sum(segmented_class == 1) and np.sum(segmented_class == 2) > np.sum(segmented_class == 3):
        return "unsafe", "knife"
    elif np.sum(segmented_class == 3) > np.sum(segmented_class == 1) and np.sum(segmented_class == 3) > np.sum(segmented_class == 2):
        return "unsafe", "shuriken"
    else:
        return "unsafe", "unknown"

# Test the combined model on an image
test_image_path = '/content/drive/MyDrive/dataset/test/knife/B0008_0017.png'
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
result, object_type = classify_and_segment(test_image)
print(f"Result: {result}, Object Type: {object_type}")
