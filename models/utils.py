import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image

# Function to load and preprocess images for prediction
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for model input.
    """
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to visualize segmentation output (for ViT)
def visualize_segmentation(input_image, segmentation_map, save_path="segmentation_result.png"):
    """
    Visualize the segmentation map on top of the original input image.
    """
    plt.figure(figsize=(10, 10))
    input_image = np.array(input_image)
    plt.imshow(input_image)
    plt.imshow(segmentation_map, alpha=0.5, cmap="jet")  # Overlay segmentation
    plt.axis("off")
    plt.savefig(save_path)
    plt.show()

# Function to evaluate detection performance (for YOLOv8)
def evaluate_detection(detections, ground_truths, iou_threshold=0.5):
    """
    Evaluate detection results based on IoU (Intersection over Union).
    """
    correct_detections = 0
    total_detections = len(detections)
    
    for det in detections:
        for gt in ground_truths:
            iou = compute_iou(det, gt)
            if iou >= iou_threshold:
                correct_detections += 1
                break
    precision = correct_detections / total_detections if total_detections > 0 else 0
    return precision

def compute_iou(detection, ground_truth):
    """
    Compute Intersection over Union (IoU) between a detection box and ground truth.
    """
    x1, y1, w1, h1 = detection
    x2, y2, w2, h2 = ground_truth
    
    # Compute the coordinates of the intersection box
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)
    
    intersection_area = w_intersection * h_intersection
    detection_area = w1 * h1
    ground_truth_area = w2 * h2
    
    union_area = detection_area + ground_truth_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

# Function to convert tensor predictions to label
def tensor_to_label(tensor, class_names):
    """
    Convert model tensor outputs to corresponding labels.
    """
    predicted_class = torch.argmax(tensor, dim=1).item()
    return class_names[predicted_class]

# Function to display a batch of images (useful for visualization)
def display_image_batch(images, labels, batch_idx=0, num_images=4):
    """
    Display a batch of images and their corresponding labels.
    """
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[batch_idx + i].transpose(1, 2, 0))  # Convert CHW -> HWC
        plt.title(f"Label: {labels[batch_idx + i]}")
        plt.axis("off")
    plt.show()

# Example function to save models
def save_model(model, model_name="model.pt"):
    """
    Save a model's state dictionary.
    """
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as {model_name}")

# Function to load models
def load_model(model, model_name="model.pt"):
    """
    Load a model's state dictionary.
    """
    model.load_state_dict(torch.load(model_name))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {model_name}")
    return model
