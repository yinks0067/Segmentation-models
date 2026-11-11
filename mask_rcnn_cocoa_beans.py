#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf


# In[5]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw


# In[6]:


from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize


# In[ ]:





# In[119]:


#Configurations


# In[7]:


#MaskRCNN config

from mrcnn.config import Config
from mrcnn import model as modellib

class BeansConfig(Config):
    # Give the configuration a recognizable name
    NAME = "beans"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # Background + number of classes you want to segment

# Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32


    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 100  # Adjust based on the size of your dataset
    VALIDATION_STEPS = 5  # Adjust based on the size of your validation dataset

    # Hardware configuration
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8  # Depending on your GPU memory size

config = BeansConfig()
config.display()


# In[121]:


#Dataset


# In[8]:


from mrcnn import utils
import json
import numpy as np
import os
import skimage.draw

class BeansDataset(utils.Dataset):
    def load_beans(self, dataset_dir, subset):
        # Define class names with descriptive names as keys
        class_names = {
            "Broken": 1,
            "Damaged": 2,
            "Fermented": 3,
            "Fraction": 4,
            "Mouldy": 5,
            "Unfermented": 6,
            "Whole": 7
        }

        # Add classes using descriptive names and integer IDs
        for class_name, class_id in class_names.items():
            self.add_class("beans", class_id, class_name)

        subset_image_dir = os.path.join(dataset_dir, "images", subset)
        subset_annotation_dir = os.path.join(dataset_dir, "annotations", subset)

        for class_name, class_id in class_names.items():
            class_image_subdir = os.path.join(subset_image_dir, class_name)
            class_annotation_subdir = os.path.join(subset_annotation_dir, class_name)

            if not os.path.isdir(class_image_subdir) or not os.path.isdir(class_annotation_subdir):
                print(f"Directory not found for class {class_name} in subset {subset}")
                continue

            for filename in os.listdir(class_image_subdir):
                if not filename.lower().endswith(".jpg"):
                    continue

                image_path = os.path.join(class_image_subdir, filename)
                annotation_filename = filename.replace(".jpg", ".json")
                annotation_path = os.path.join(class_annotation_subdir, annotation_filename)

                if os.path.exists(annotation_path):
                    with open(annotation_path) as f:
                        annotations = json.load(f)
                    image_info = next((item for item in annotations['images'] if item['file_name'].lower() == filename.lower()), None)
                    if image_info:
                        self.add_image("beans",
                                       image_id=image_info['id'],
                                       path=image_path,
                                       width=image_info['width'],
                                       height=image_info['height'],
                                       annotations=annotations['annotations'])
                else:
                    print(f"Annotation file {annotation_filename} not found for image {filename}.")


        
        
        
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        masks = np.zeros([info['height'], info['width'], len(annotations)], dtype=bool)  # Updated dtype
        class_ids = np.array([annotation['category_id'] for annotation in annotations])
    
        for i, annotation in enumerate(annotations):
            segmentation = annotation['segmentation'][0]
            x_points = np.array(segmentation[::2])
            y_points = np.array(segmentation[1::2])
            x_points = np.clip(x_points, 0, info['width'] - 1)
            y_points = np.clip(y_points, 0, info['height'] - 1)
            rr, cc = skimage.draw.polygon(y_points, x_points)
            masks[rr, cc, i] = True

        return masks.astype(np.bool_), class_ids.astype(np.int32)  # Updated np.bool_ here as well
        
        
        

# Setup dataset paths
dataset_dir = 'C:\\Users\\user\\Documents\\datasets'

# Initialize and prepare datasets
dataset_train = BeansDataset()
dataset_train.load_beans(dataset_dir, 'train')
dataset_train.prepare()

dataset_test = BeansDataset()
dataset_test.load_beans(dataset_dir, 'test')
dataset_test.prepare()

print("Training and test datasets are ready.")



# In[8]:


#Testing


# In[9]:


# Output how many images were loaded for each subset
print("Number of train images: ", len(dataset_train.image_ids))
print("Number of test images: ", len(dataset_test.image_ids))


# In[7]:


from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import random

# Assuming dataset_train is already defined and prepared
def visualize_image_and_mask(dataset, image_id):
    """Visualizes an image and its masks from the dataset."""
    # Load image and mask
    image = io.imread(dataset.image_info[image_id]['path'])
    mask, class_ids = dataset.load_mask(image_id)

    # If no mask or class_ids, display image only
    if mask is None or class_ids is None:
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title("Image (No masks found)")
        plt.axis('off')
        plt.show()
        return

    # Create a mask display of all instances by summing
    all_masks = np.sum(mask, axis=-1)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    ax[0].imshow(image)
    ax[0].axis('off')
    ax[0].set_title("Original Image")

    ax[1].imshow(image)
    ax[1].imshow(all_masks, alpha=0.5, cmap='jet')
    ax[1].axis('off')
    ax[1].set_title("Image with Mask Overlay")

    plt.show()

# Example usage
image_ids = dataset_train.image_ids
random_image_id = random.choice(image_ids)
visualize_image_and_mask(dataset_train, random_image_id)


# In[8]:


#create Model


# In[9]:


model_dir = "C:\\Users\\user\\Desktop\\logs"


# In[10]:


weights_path = "C:\\Users\\user\\Documents\\Mask-RCNN-TF2\\kangaroo-transfer-learning\\mask_rcnn_coco.h5"
print(f"COCO weights path: {weights_path}")
print("Exists:", os.path.exists(weights_path))


# In[11]:


model = modellib.MaskRCNN(mode="training", config=config, model_dir=model_dir)


# In[78]:


#Training log
get_ipython().run_line_magic('load_ext', 'tensorboard')



# In[79]:


get_ipython().run_line_magic('reload_ext', 'tensorboard')


# In[80]:


get_ipython().run_line_magic('tensorboard', '--logdir "C:\\\\Users\\\\user\\\\Desktop\\\\logs"')


# In[12]:


# Training configuration
print("Training network heads")
history=model.train(dataset_train, dataset_test,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers='heads')


# In[82]:


#Model Evaluation


# In[ ]:





# In[10]:


from mrcnn.config import Config

class InferenceConfig(BeansConfig):
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # Background + number of classes you want to segment

# Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32


    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 100  # Adjust based on the size of your dataset
    VALIDATION_STEPS = 5  # Adjust based on the size of your validation dataset

    # Hardware configuration
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # Depending on your GPU memory size
    BATCH_SIZE = 1 

inference_config = InferenceConfig()
inference_config.display()
from mrcnn.config import Config
print("Batch size for inference:", inference_config.BATCH_SIZE)  # Should print 1

     


# In[11]:


import os
import numpy as np
import skimage.io
from mrcnn import model as modellib


MODEL_DIR = 'C:\\Users\\user\\Desktop\\logs'

model_inference = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
weights_path = 'C:\\Users\\user\\Desktop\\logs\\mask_rcnn_beans_0010.h5'
model_inference.load_weights(weights_path, by_name=True)



# In[ ]:


#computing IOU and accuracy


# In[20]:


import numpy as np

# Function to compute IoU
def compute_iou(gt_box, pred_box):
    xA = max(gt_box[0], pred_box[0])
    yA = max(gt_box[1], pred_box[1])
    xB = min(gt_box[2], pred_box[2])
    yB = min(gt_box[3], pred_box[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    boxBArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Function to compute bounding boxes from masks
def compute_bboxes_from_mask(masks):
    num_instances = masks.shape[-1]
    boxes = np.zeros([num_instances, 4], dtype=np.int32)
    for i in range(num_instances):
        m = masks[:, :, i]
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            x2 += 1
            y2 += 1
        else:
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes

ious = []
predictions = []
gt_labels = []

for image_id in dataset_test.image_ids:
    image = dataset_test.load_image(image_id)
    gt_mask, gt_class_id = dataset_test.load_mask(image_id)
    gt_box = compute_bboxes_from_mask(gt_mask)  # Compute bounding boxes from mask
    r = model_inference.detect([image], verbose=0)[0]

    for i, pred_box in enumerate(r['rois']):
        score = r['scores'][i]
        pred_class_id = r['class_ids'][i]
        
        if score > 0.5:  # threshold for detection confidence
            predictions.append(pred_class_id)
            gt_labels.append(gt_class_id[0] if gt_class_id.size > 0 else 0)  # No object as class 0

            # Calculate IoU for each prediction with corresponding ground truth
            if gt_class_id.size > 0:  # ensure there is a ground truth to compare
                # Calculate IoU for the first detected box and the first ground truth box
                current_iou = compute_iou(gt_box[0], pred_box)
                ious.append(current_iou)

average_iou = np.mean(ious) if ious else 0
print(f"Average IoU: {average_iou}")

from sklearn.metrics import confusion_matrix, accuracy_score

# Calculate confusion matrix and accuracy
conf_matrix = confusion_matrix(gt_labels, predictions)
accuracy = accuracy_score(gt_labels, predictions)

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)



# In[ ]:


#plotting confusion matrix


# In[22]:


get_ipython().system('pip install seaborn')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample data from your confusion matrix calculation
conf_matrix = np.array([[253]])  # Update this with your actual confusion matrix data if needed

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', square=True,
            xticklabels=['Predicted Label'], yticklabels=['True Label'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:


class_labels = ['Class 1', 'Class 2', 'Class 3', ..., 'Class N']  # Replace with your actual class labels
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', square=True,
            xticklabels=class_labels, yticklabels=class_labels)


# In[ ]:


#testing on random image


# In[12]:


from mrcnn import visualize 
import random

# Choose random images to test
image_ids = np.random.choice(dataset_test.image_ids, 10)

for image_id in image_ids:
    # Load image
    original_image = dataset_test.load_image(image_id)

    # Detect objects
    results = model_inference.detect([original_image], verbose=8)[0]

    # Visualize results
    visualize.display_instances(original_image, results['rois'], results['masks'],
        results['class_ids'], dataset_test.class_names, results['scores'])


# In[ ]:





# In[142]:


# Display results for specified classes
# Ensure your dataset_test has the correct class names mapped
dataset_test.class_names = ['BG', 'Broken', 'Damaged', 'Fermented', 'Unfermented']

# Run detection and visualize results for each image
for image_id in dataset_test.image_ids:
    # Load the image
    image = dataset_test.load_image(image_id)
    
    # Run detection
    result = model.detect([image], verbose=1)[0]
    
    # Visualize results for this image
    visualize.display_instances(image, result['rois'], result['masks'], 
                                result['class_ids'], dataset_test.class_names, 
                                result['scores'])


# In[ ]:


#Hyper parameter tunning


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




