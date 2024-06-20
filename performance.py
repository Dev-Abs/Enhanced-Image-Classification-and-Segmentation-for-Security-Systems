import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Constants
IMAGE_HEIGHT = 256  # Image height for segmentation
IMAGE_WIDTH = 256  # Image width for segmentation

# Path to test data and annotations
test_folder = '/content/drive/MyDrive/dataset/test'
annotations_folder = '/content/drive/MyDrive/dataset/test/annotations_2'

# Initialize lists for storing true labels and predictions
true_labels = []
pred_labels = []

# Initialize variables for Dice coefficient calculation
dice_scores = []

# Iterate through categories
for category in ['safe', 'GUN', 'knife', 'shuriken']:
    predicted_masks_folder = os.path.join(test_folder, category, 'predicted_masks')

    if category != 'safe':  # 'safe' category does not have segmentation masks
        true_category_folder = os.path.join(annotations_folder, category)

        for filename in os.listdir(predicted_masks_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                pred_mask_path = os.path.join(predicted_masks_folder, filename)
                true_mask_path = os.path.join(true_category_folder, filename)

                # Read predicted and true masks
                pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
                true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)

                if pred_mask is None or true_mask is None:
                    continue

                true_mask = cv2.resize(true_mask, (IMAGE_WIDTH, IMAGE_HEIGHT))
                true_mask = (true_mask > 127).astype(np.uint8)
                pred_mask = (pred_mask > 0.5).astype(np.uint8)

                # Calculate Dice coefficient
                intersection = np.sum(true_mask * pred_mask)
                dice_score = (2. * intersection) / (np.sum(true_mask) + np.sum(pred_mask))
                dice_scores.append(dice_score)

                # Plot true and predicted masks
                # plt.figure(figsize=(10, 5))
                # plt.subplot(1, 2, 1)
                # plt.title('True Mask')
                # plt.imshow(true_mask, cmap='gray')
                # plt.subplot(1, 2, 2)
                # plt.title('Predicted Mask')
                # plt.imshow(pred_mask, cmap='gray')
                # plt.show()

                # Append true and predicted labels
                true_labels.append(category)
                pred_labels.append(category if dice_score > 0.5 else 'unknown')

    # For 'safe' category
    else:
        safe_images_folder = os.path.join(test_folder, 'safe')
        for filename in os.listdir(safe_images_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                true_labels.append('safe')
                pred_labels.append('safe')  # Assuming all images in 'safe' folder are predicted correctly

# Calculate and print performance metrics
overall_accuracy = accuracy_score(true_labels, pred_labels)
conf_matrix = confusion_matrix(true_labels, pred_labels)
dice_coefficient = np.mean(dice_scores) if dice_scores else 0

print(f"Overall Accuracy: {overall_accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Dice Coefficient: {dice_coefficient}")
print(classification_report(true_labels, pred_labels, target_names=['safe', 'GUN', 'knife', 'shuriken', 'unknown']))
