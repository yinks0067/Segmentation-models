Predicting Cocoa Bean Shelf Life with Deep Learning
Cocoa beans lose value quickly after harvest if they're broken, mouldy or badly fermented, and grading them is still mostly done by eye. For my MSc Data Science dissertation at the University of Salford (Distinction), I asked whether a computer vision model could look at a photo of cocoa beans and tell which condition each bean is in, as a step towards predicting how long they will keep.

I compared two segmentation models, Mask R-CNN and U-Net, and this repo has the code for both.

File
What it is
Unet model - Cocoa Bean Segmentation.ipynb
The U-Net notebook, including tuning and the optimiser comparison
mask_rcnn_cocoa_beans.py
The Mask R-CNN training and evaluation script

Results
Model
Accuracy
IoU
Mask R-CNN
97%
0.07
U-Net, first run (Adam, batch 8, 50 epochs)
67.1%
0.60
U-Net, tuned (Adam, batch 16, 48 epochs)
77.7%
0.62


The most important result isn't the 77.7%. It's the Mask R-CNN row. Its accuracy looked excellent, but its IoU of 0.07 showed it was barely finding the beans in the right place. Accuracy alone would have picked the wrong model. Judging both models on IoU and confusion matrices is what showed U-Net was the better choice.

The two IoU figures aren't measured in quite the same way. For U-Net it compares predicted and true masks pixel by pixel. For Mask R-CNN it compares bounding boxes. The gap between them is still far too big to be explained by that.
The data
7,428 images of cocoa beans on a white background, collected at a factory in South Sulawesi, Indonesia by Yudhi Adhitya (Fukuoka University)
7 classes: whole bean (1,187), unfermented (1,776), mouldy (1,255), broken (1,046), fermented (916), skin damage (822) and bean fraction (426)
I annotated the images myself, using Apeer for the U-Net masks and MakeSense for the Mask R-CNN polygons
How I built the Mask R-CNN
Annotated the beans as polygons in MakeSense and exported them as COCO-style JSON
Wrote a custom dataset loader that reads each class folder, matches every image to its annotation file and turns the polygons into masks
Used transfer learning: started from weights pretrained on the COCO dataset and trained only the head layers for 10 epochs, with 128 x 128 images and anchor sizes scaled down for small objects
Evaluated it on the test set with IoU, accuracy and a confusion matrix, and drew the predicted masks on random test images to check them by eye
Tracked training with TensorBoard

Looking back at the evaluation. When I reread this code I found two things I'd do differently today. The accuracy compared every prediction in an image with only the first true label in that image, and the confusion matrix ended up with a single cell. That means the 97% says very little about telling the classes apart, which matches the low IoU. A fair rerun would score each prediction against the true bean it overlaps most, and measure IoU on the masks rather than the boxes. I've left the code as it was submitted so the results above match my dissertation.
How I built the U-Net
Prepared the images: converted them to greyscale, resized them to 256 x 256 and normalised the pixel values
Encoded the masks so each pixel belongs to one of 8 classes (the 7 bean types plus background)
Split the data into 80% training and 20% testing
Built a multi-class U-Net in Keras with about 1.9 million trainable parameters, trained with categorical cross-entropy
Evaluated every run with accuracy, IoU and a confusion matrix, and compared the predicted masks with the true masks by eye
Tuned it by searching over learning rate (0.0001 to 0.0005), batch size and epochs. Batch size 16 and 48 epochs gave the best model.
Tried other optimisers to see if Adam could be beaten:

Optimiser
Best accuracy
Adam
77.7%
RMSprop
70.1%
AdaGrad
70.1%
SGD
70.1%


RMSprop, AdaGrad and SGD all stopped at exactly the same accuracy, and their validation accuracy stayed flat during training. That pattern usually means a model has fallen back to predicting the most common class (here, the background) for most pixels. So the honest reading is that Adam was the only optimiser in these runs that actually learned to separate the bean classes well.
What I learned
Pick the metric that matches the problem. For segmentation, IoU tells you far more than accuracy.
Look at the predictions, not just the score. The confusion matrix showed the model was strong on mouldy and skin-damaged beans but often mixed up similar classes, such as broken beans and whole beans.
Tuning matters. Changing batch size and epochs took U-Net from 67% to 77.7% with no change to the architecture.
Limitations and next steps
The classes are imbalanced (1,776 unfermented beans against 426 bean fractions), which hurts the rarer classes. Class weights or more augmentation would help.
Library version clashes in Google Colab cost time and limited how much tuning I could do on Mask R-CNN. It was only trained for 10 epochs, on its head layers.
The Mask R-CNN evaluation should be redone as described above before its numbers are compared directly with U-Net's.
Next I'd try a pretrained encoder (for example a ResNet backbone) and report precision and recall for each class.
Tech
Python · Keras / TensorFlow · Mask R-CNN (Matterport, TF2) · OpenCV · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter
Running it
Open the U-Net notebook in Jupyter or Google Colab and point the image and mask paths at your copy of the dataset. The Mask R-CNN script needs the Matterport Mask R-CNN library (TensorFlow 2 version) and the COCO weights file mask_rcnn_coco.h5. Set the dataset, log and weights paths at the top of the script, or as the environment variables named there. The dataset isn't included here because it belongs to its original authors.



I'm Yinka, an AI Solutions Engineer and data scientist in Manchester. More of my work is on my GitHub profile, and you can find me on LinkedIn.
