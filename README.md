# FaceMask Detection and Segmentation 

## 1. Introduction
This project focuses on identifying and segmenting face masks in images using a combination of deep learning and machine learning techniques. It involves region-based segmentation with traditional image processing methods, mask segmentation using U-Net, and binary classification using both CNN models and handcrafted features with conventional classifiers. The aim is to compare the accuracy and effectiveness of these different approaches for mask detection and segmentation.

### Contributers:

(IMT2022071) Samyak Jain <Samyak.jain@iiitb.ac.in>


(IMT2022118) Shashwat Chaturvedi <Shashwat.chaturvedi@iiitb.ac.in>


(IMT2022520) Pupur Natil <Nuput.Patil@iiitb.ac.in>   <<<<-----


---

## 2. Dataset

### Source:

The datasets used in this project come from the following sources:  


- [Face Mask Detection Dataset](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)  

- [MFSD Dataset](https://github.com/sadjadrz/MFSD)  



```
dataset
├── with_mask # contains images with mask
└── without_mask # contains images without face-mask
```
```
MSFD
├── 1
│   ├── face_crop # face-cropped images of images in MSFD/1/img
│   ├── face_crop_segmentation # ground truth of segmend face-mask
│   └── img
└── 2
    └── img
```

### Dataset Structure:
- **Training Set:** Images used for training the models.
- **Testing Set:** Images used to evaluate model performance.
- **Annotations:** Mask region labels for segmentation tasks.

---


## 3. Objectives and Evaluation  

### a. Binary Classification with Handcrafted Features and ML Models (4 Marks)  
1. Extract important handcrafted features from facial images.  
2. Train and test at least two machine learning models, such as SVM and Neural Networks, to classify images as "with mask" or "without mask."  
3. Evaluate and compare the accuracy of these models to determine the best-performing classifier.  

### b. Binary Classification with CNN (3 Marks)  
1. Design and train a Convolutional Neural Network (CNN) for mask classification.  
2. Experiment with different hyperparameters, including learning rate, batch size, optimizer, and activation functions, and report the findings.  
3. Compare the CNN’s performance with the traditional machine learning classifiers.  

### c. Region-Based Segmentation Using Traditional Methods (3 Marks)  
1. Apply classical segmentation techniques, such as thresholding and edge detection, to identify and segment mask regions in images classified as "with mask."  
2. Visualize the segmented regions and assess their accuracy.  

### d. Mask Segmentation with U-Net (5 Marks)  
1. Train a U-Net model to accurately segment face masks from images.  
2. Compare its segmentation performance with traditional methods using metrics like Intersection over Union (IoU) and Dice Score.  



---



## 5. Results  

### Evaluation Metrics  
- **For Classification:** Accuracy, Precision, Recall, and F1-score  
- **For Segmentation:** Intersection over Union (IoU) and Dice Score  

| Model | Accuracy (%) | IoU | Dice Score |  
|-------------------------|--------------|------|------------|  
| SVM (Binary Classification) | 88.04780876494024% (80-20 split) | - | - |  
| MLP (Binary Classification) | 92.56308100929614% (80-20 split) | - | - |  
| CNN | 96.31% | - | - |  
| Region-Growing (Segmentation) | - | 0.3997 (Avg) | 0.5365 (Avg) |  
| Otsu's Thresholding (Segmentation) | Provided below(5) | - | - |  
| U-Net (Mask Segmentation) | *[Results pending – To be updated Sassy]* | - | - |  



---


## 6. Observations & Analysis

### Part_a: Binary Classification Using Handcrafted Features and ML Classifiers

##### Support Vector Machine (SVM)
- The SVM model performed **fairly well**, achieving an accuracy of **88.05%**.
- It maintained a good balance between **precision and recall**, meaning it didn’t favor one class over the other.
- The **f1-score of 0.88** suggests that the model makes reliable predictions but still has some room for improvement.
- A few misclassifications could be due to **lighting variations, occlusions, or feature extraction limitations**.
- Overall, SVM does a solid job, but it might not be the best option for complex patterns.

##### Multi-Layer Perceptron (MLP)
- The MLP model performed **better than SVM**, with an accuracy of **92.56%**.
- Its **f1-score of 0.93** shows that it captures patterns more effectively.
- Compared to SVM, MLP had **fewer false negatives**, meaning it was better at correctly identifying both masked and unmasked faces.
- The improvement is likely because MLP, being a neural network, can learn more **complex relationships** in the data.
- With some fine-tuning (like adjusting hidden layers or activation functions), it could perform even better.

##### Key Takeaways:
- **Handcrafted features (HOG + LBP) were effective**
- For further improvements, switching to a **CNN-based feature extraction approach** might be the next step.


### Part_b: Using CNN

#####  CNN   
✅ With CNN, we saw a **huge improvement**, achieving **96.3% accuracy**, making it clear that deep learning is far more effective for this task.

#####  Accuracy Trends
- The model **quickly hit 100% training accuracy**, but validation accuracy leveled off at **~96%**.  
- This suggests **overfitting**—the model memorized the training data well but didn’t generalize perfectly to new images.  
- Still, **96% validation accuracy** is a solid performance!

#####  Test Performance  
- **Final test accuracy:** **96.3%**—way better than traditional methods.  
- **Breakdown of results:**  
  - "With Mask" → **97% Precision, 96% Recall**  
  - "Without Mask" → **96% Precision, 96% Recall**  
- This means the model is **equally good** at classifying both categories, with no major bias.

#### Hyperparameters and Experiments

The following hyperparameters and experimental settings were used in the code:

1. **Image Resizing**: The images are resized to 64x64 pixels to match the input size required by the Convolutional Neural Network (CNN).
   ```python
   img = cv2.resize(img, (64, 64))  # Resize to match CNN input

2. **Normalization**: The pixel values of the images are normalized to the range [0, 1] by dividing by 255.0
   ```python
   X = np.array(X, dtype=np.float32) / 255.0  # Normalize pixel values

4. **Train-Test Split**: The dataset is split into training and testing sets with a test size of 20% and a random state of 42 for reproducibility.
   ```python
   train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42)


### Part_c: Segmentation using traditionals
##### **1. Segmentation Performance**
| Method                | IoU Score | Dice Score |
|-----------------------|----------|------------|
| **Region Growing**    | 0.3997   | 0.5365     |
| **Otsu’s Thresholding** | 0.3156   | 0.4430     |

##### **Region Growing Method**
- Achieved **higher accuracy**, with an IoU of **0.3997** and a Dice score of **0.5365**.
- Adaptive to local intensity variations but **sensitive to tolerance selection**.
- Works well on images with smooth transitions but may leak into adjacent regions.

##### **Otsu’s Thresholding**
- Lower IoU of **0.3156** and Dice score of **0.4430**.
- Global thresholding leads to **over-segmentation or under-segmentation** in non-uniform lighting conditions.
- Computationally efficient but **not adaptive** to local intensity variations.

#### **2. Key Takeaways**
- **Region Growing is more accurate** but requires fine-tuning of tolerance.
- **Otsu’s method is faster**, but struggles in varying lighting conditions.
- **Potential Improvements:**
  - Experimenting with **K-Means clustering** for more adaptive segmentation.
  - **Hybrid approaches**, such as Otsu + Morphological processing, may enhance results.
  - Using **deep learning-based segmentation** for further accuracy improvement.

#### **3. Conclusion**
Region Growing currently performs better for this dataset, but further tuning and hybrid approaches could yield even better results.






## 7. How to Run the Code

### Setup

1. First, clone this repository and navigate into the project directory:
   ```bash
   git clone https://github.com/JConquers/VR_Project_1
   cd VR_PROJECT_1

   ```
2. Set up a virtual environment and install the required dependencies:
    
   ```bash
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt

   
3. Download the dataset as specified and ensure the dataset and MSFD directories are placed at the same level as the repository. Additionally, create a directory named output using the command mkdir output. Your final folder structure should resemble this: ```
    ```
    .
    ├── dataset
    ├── MSFD
    ├── output
    ├── scripts
    └── images
    ```
    #### Other files like README.md, pdf, etc are not shown in this tree.

5. Execute the scripts:

   ##### Inside the scripts directory, you’ll find two Jupyter notebooks: part_a_b.ipynb and part_c_d.ipynb. These scripts can be executed together or separately to observe partial results.
    ---
























































































