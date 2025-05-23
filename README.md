
# Executive Project Summary
## Problem statement: AI-Powered Imaging Technology for Enhancing Dental Examinations

Fear of visiting the dentist is a common issue, especially among children and elderly patients. The intimidating dental environment, filled with complex equipment, bright lights, and sharp instruments, can discourage individuals from seeking timely dental care. Additionally, conventional dental examinations require healthcare professionals to manually inspect each tooth, which is time-consuming, can cause discomfort, and may lead to missed early signs of decay. Current diagnostic tools such as X-rays, including Orthopantomograms (OPG) and bitewing X-rays, offer effective imaging solutions but come with drawbacks. They require scheduled appointments, can be costly, and may not always be covered by insurance. These limitations highlight the need for an alternative solution that enhances efficiency, improves patient comfort, and reduces diagnostic errors. This project aims to introduce AI-powered imaging technology as a transformative approach to dental examinations. High-resolution images of the oral cavity, captured from multiple angles (occlusal, buccal, and lingual), will be analyzed using machine learning algorithms such as YOLO for object detection. This approach will allow accurate identification of cavities and early signs of decay without invasive procedures or traditional probing methods. 
Keywords:  AI Imaging Technology, YOLO, Dental Examination, Public Dataset 

# Research Question
How can AI-powered imaging technology enhance dental examinations by improving accuracy, reducing patient discomfort, and minimizing reliance on traditional X-ray imaging while maintaining cost-effectiveness and efficiency? 
 
## Project goals: 

Using AI-powered imaging technology to modify dental examinations. High-resolution images of the oral cavity will be captured from various angles (occlusal, buccal, and lingual) to provide detailed views of each tooth. These images will be processed using machine learning algorithms, such as YOLO for object detection, to accurately identify cavities or early signs of decay without time-consuming procedures involving probing or mirrors. 

- Improve accuracy in detecting dental problems, reducing the chances of missing cavities.   
- Provide a cost-effective alternative to X-rays by eliminating the need for expensive imaging equipment and associated fees.   
- Improve the overall patient experience by minimizing discomfort and stress.   
- Offer ethical, non-invasive, and safe methods for dental examinations.   
- It reduces the need for a lot of sterilization equipment and instruments in dental hospitals, making the process simpler and easier to manage. 

## Project Merit: 

This project will benefit both patients and healthcare professionals. For patients who are scared of dental treatments, especially kids and elderly people, this technology can help reduce fear by making the examination process faster and more comfortable. It can encourage more people to visit the dentist without hesitation. For hospitals and clinics, it reduces the need for expensive equipment like X-ray machines and reduces the burden of sterilizing and buying multiple instruments. This can save time, money, and resources while improving the overall quality of care. 

## Project Timeline (3 Months) 

This project follows a structured timeline to ensure efficient execution and completion within three months.

![gannt chart](https://github.com/user-attachments/assets/6651dfdd-f52e-4d41-89e9-a84d5d5bc4ba)

| Task                                      | Start Date  | End Date    | Duration  |
|-------------------------------------------|------------|------------|-----------|
| Data Collection & Permission Requests    | Feb 1, 2024 | Feb 15, 2024 | 2 weeks  |
| Data Preprocessing & Annotation          | Feb 16, 2024 | Feb 29, 2024 | 2 weeks  |
| Model Training & Optimization (YOLOv8)   | Mar 1, 2024  | Mar 20, 2024 | 3 weeks  |
| Validation & Performance Analysis        | Mar 21, 2024 | Apr 5, 2024  | 2 weeks  |
| Comparative Analysis with Traditional Methods | Apr 6, 2024  | Apr 15, 2024 | 1.5 weeks |
| Final Report & Presentation Preparation  | Apr 16, 2024 | Apr 30, 2024 | 2 weeks  |

## Risk List

As with any project, several risks could impact the successful completion of this AI-powered dental imaging study. 

| Risk Name (Value)                                   | Impact | Likelihood | Description |
|-----------------------------------------------------|--------|------------|-------------|
| Limited Access to Dental Imaging Data (72)         | 9      | 8          | Lack of access to real-world dental image datasets could limit model training and validation. Mitigation: Seek permissions from dental schools and use Kaggle datasets. |
| Technical Challenges in YOLOv8 Model Training (56)  | 8      | 7          | The complexity of deep learning model optimization could lead to inefficiencies or errors. Mitigation: Conduct regular testing, debugging, and parameter tuning. |
| Computational Resource Limitations (42)            | 7      | 6          | Google Colab's memory constraints and session limits may slow down model training. Mitigation: Use Colab Pro, optimize datasets, and train in smaller batches. |
| Time Constraints and Tight Deadlines (54)          | 6      | 9          | The project must be completed in three months, increasing the risk of delays. Mitigation: Follow a structured timeline with clear milestones. |
| Team Availability and Coordination Issues (35)     | 5      | 7          | Scheduling conflicts may arise among team members. Mitigation: Maintain consistent communication, set clear responsibilities, and ensure regular progress check-ins. |

## Literature Review

Artificial Intelligence (AI) has revolutionized dental caries diagnosis by improving accuracy and efficiency through machine learning (ML) and deep learning (DL) techniques. Traditional methods like visual-tactile examination and radiographic imaging often struggle with early detection and subjectivity, leading to diagnostic inconsistencies (Selwitz et al., 2007). AI-driven approaches, particularly Convolutional Neural Networks (CNNs) and Support Vector Machines (SVMs), have shown superior accuracy in analyzing dental images, identifying caries with over 90% precision (Yamashita et al., 2018). AI-powered tools such as AssistDent and YOLOv3-based object detection models have further improved real-time detection, reducing false positives and increasing diagnostic efficiency (Ding et al., 2021). However, challenges such as data privacy concerns, model bias, and clinician acceptance remain critical obstacles to widespread adoption (Rischke et al., 2022). Future advancements in federated learning, tele-dentistry integration, and explainable AI (XAI) are expected to refine AI-driven diagnostics, making them more accessible and reliable for dental professionals.  

Recent advancements in AI-driven dental imaging have demonstrated promising results in improving diagnostic accuracy and efficiency. AbuSalim et al. (2024) introduced a multi-granularity approach using YOLO-based object detection models for effective tooth detection and classification. Their findings suggest that AI-powered imaging can significantly enhance diagnostic precision, reducing errors associated with manual inspections. Furthermore, Ramírez-Pedraza et al. (2025) explored deep learning applications in oral hygiene, specifically focusing on automated dental plaque detection using the YOLO framework. Their study highlights the potential of AI models in improving preventive dental care by offering real-time detection and quantification of dental plaque using the O’Leary Index. These advancements indicate a growing trend toward AI integration in dentistry, paving the way for innovative solutions that minimize patient discomfort while optimizing clinical workflows. 

AbuSalim et al. (2024) explore the application of YOLO-based object detection models for dental image analysis, specifically focusing on tooth detection and classification. The study addresses the limitations of traditional dental imaging techniques and evaluates the effectiveness of AI-driven solutions. The researchers investigate whether YOLO-based object detection can enhance tooth classification accuracy compared to conventional methods. Their methodology involves training a YOLOv5 model on clinically sourced dental images, preprocessing the data, and evaluating model performance using standard accuracy metrics such as mean Average Precision (mAP). The findings indicate that AI-based object detection significantly improves diagnostic accuracy, reducing errors commonly associated with manual inspections. The study demonstrates that YOLO models can efficiently identify multiple teeth with high precision, highlighting the potential for real-time AI-assisted dental analysis. However, the research also notes certain limitations, such as the relatively small dataset size, which may affect generalizability, and the dependency on high-quality image preprocessing for optimal performance. The study contributes to the field by demonstrating the feasibility of AI-powered dental diagnostics and emphasizing the importance of dataset quality in model training. This research is highly relevant to the current project, as it provides a strong foundation for leveraging YOLO-based models in dental imaging. Building on these findings, our study will extend this work by utilizing YOLOv8, incorporating a more diverse dataset, and refining detection accuracy and efficiency for practical clinical applications.

The study by Salahin et al. (2023) focuses on using smartphone images to detect cavities in teeth. It uses the YOLOv5 model for object detection because it is fast and accurate. The model creates feature maps in three different sizes to detect small, medium, and large objects in the images. To see how well the model performs, the study measures accuracy using mean average precision (mAP) by checking true positive and false positive results. The results show that YOLOv5 works well for detecting medium and large cavities but is slightly less accurate with small ones. This research shows that smartphone images can be a simple and affordable way to screen for cavities.

This project proposes the development and implementation of an AI-driven imaging system to modernize dental diagnostics. The specific objectives are to improve accuracy by utilizing machine learning-based object detection to enhance early cavity detection and reduce missed diagnoses. It aims to reduce costs by providing a cost-effective alternative to traditional X-rays, minimizing the need for expensive imaging equipment and additional fees. The project seeks to enhance the patient experience by offering a non-invasive, stress-free examination method, especially for patients with dental anxiety. Additionally, it ensures ethical and safe practices by reducing reliance on radiation-based diagnostics and minimizing discomfort through advanced imaging technology. Finally, it simplifies dental clinic operations by decreasing the need for excessive sterilization equipment, making the workflow more efficient and reducing hospital overhead costs. 

#### Keywords:  AI Imaging Technology, YOLO, Dental Examination, Public Dataset 

## Methodology
This study will build upon previous research but integrate the latest advancements in object detection, specifically utilizing YOLOv8 for multi-granularity tooth analysis and automated dental plaque detection. The research model will follow a data-driven approach, leveraging high-resolution intraoral images to train and validate AI models for tooth classification, cavity detection, and plaque quantification. The methodology will include data collection from diverse dental case studies, preprocessing and augmentation of images, and iterative model training and validation to ensure robustness and accuracy. By employing an updated YOLOv8 framework, the study will refine the detection accuracy beyond prior research models, ensuring enhanced performance in real-time clinical settings. The outcomes of this model will be validated through comparative analysis with traditional dental examination methods, ensuring the feasibility and reliability of AI-assisted diagnostics. 

## Key Steps & Methodology:

- Step 1: Collect Data  
To collect data, we start by gathering dental images from available sources. Some of these images come from public datasets, like those on Kaggle, while others may come from healthcare organizations. The dataset should include both healthy teeth images (no cavities) and cavity images (with cavities). Having a large and well-balanced dataset is important because it helps the model learn more effectively.
- Step 2: Data Processing (Data Augmentation)  
We will use data augmentation techniques to increase the size of our dataset by applying different transformations to the images. These transformations include horizontal flipping, vertical flipping, cropping, and rotation. By applying these techniques, we can create more varied images, which will help the model learn to recognize caries (cavities) in different situations. This makes the model better at generalizing, so it can work well on new, unseen images.
- Step 3: Data Processing  
We’ll use NumPy and Pandas libraries to manage and simplify the data. NumPy will help with numerical tasks like handling image pixels and performing operations on them. Pandas will be used to organize the data, making it easier to process, especially when dealing with image labels and other related information. These libraries will help make the data processing more efficient and manageable.
- Step 4: Data Splitting  
We will divide the dataset into two parts. The Training Set will have most of the images and will be used to train the model. This allows the model to learn how to recognize the differences between healthy and cavity teeth. The Validation Set will have a few images and will be used to check how well the model performs after training. It helps us see if the model is working correctly and can make accurate predictions on new images. 
- Step 5: Data Labeling  
Data labeling is really important for training the model because it helps the model understand the differences between healthy teeth and teeth with caries. We will label the images to identify whether they show healthy teeth or cavity teeth. To do this, we'll use labeling tools like LabelImg . Each image will have a label attached, such as healthy or cavity, so the model can learn from these labels and make accurate predictions when given new images.
- Step 6: Training the YOLOv8 Model  
We’ll use YOLOv8, a model created by Ultralytics, to detect cavities in dental images. At first, the model doesn’t know what cavities are. During training, it learns to recognize cavities by being shown images repeatedly and adjusting its settings (called weights) to improve its understanding. YOLOv8 offers different model sizes, such as small (YOLOv8s), medium (YOLOv8m), and large (YOLOv8l), where the small model is faster but less accurate, and the large model is more accurate but slower. Training is done in batches, where we process a small group of images at a time, and we repeat this process many times, which is called an epoch. The more epochs we run, the better the model becomes at accurately detecting cavities in the images. After completing the training, we paln to test different versions of yolo model evaluate their performance. 
- Step 7: Model Evaluation  
After training, we evaluate the model's performance using the validation set. The goal is to see how well the model can detect cavities. We use metrics like Accuracy,precision,and recall evaluates the model predicts the objects in the image. Precision determines the how many of the predicted objects are correct, while recall evaluateshow well the model identifies all actual objects in the image. we also analyze True Positive to check if the model correctly identified cavities, and False Positive to track any mistakes it made. If the model's performance isn't good enough, we go back to the training step, adjusting the parameters or training for more epochs to improve accuracy.

## Initial Technical Aspect  
| Steps	|Description |	Technical aspects |
| ---------------|----------|--------------|
|Data collection |	Collect dental images from Kaggle or healthcare organizations |	Kaggle |
|	Data processing(Augmentation) |	It is helps for making more useful images from original one |	Horizontal / vertical flip, rotation and cropping of images |
|	Data processing|	For simplifying the data set | NumPy and pandas |
|	Data splitting |	Here the data divided into two types for model training |	Randomly divide the images or using libraries |
|	Data labelling |	Give names to the image with bounding box |	Tools like Labellmg |
|	Model training	| It is for differentiating healthy teeth from cavity-affected teeth|	YOLO model like small, medium, or large epoches| 
|	Model evaluation |	It is used for accuracy of results	| Based on true positive, false positive, False negative |



## Resources Needed

| Resource                                      | Dr. Hale Needed? | Investigating Team Member | Description |
|-----------------------------------------------|------------------|---------------------------|-------------|
| Google Colab                                  | Yes              | Sravani/Daun/Reshmi       | Cloud-based platform for training and testing the YOLOv8 model. |
| YOLOv8 (Ultralytics)                          | yes              | Reshmi/Daun               | Pre-trained object detection model for dental image analysis. |
| Kaggle Dental Imaging Dataset                 | No               | Sravani                   | Publicly available dataset for training and validating the model. |
| Local Dental School Imaging Data              | No               | Daun                      | Requesting real-world dental images for model enhancement. |
| Python (NumPy, Pandas,)                       | Yes              | Reshmi/Sravani            | Required libraries for data processing, model training, and evaluation. |
| GitHub                                        | No               | Daun/Sravani/Reshmi       | Version control for project collaboration and documentation. |
| Data processing                               | No               | Sravani                   | Select the good quality of images, split them into training and validation sets and label them using tools|





# Milestone 2
## Environment setup
We have established and documented the required environments, including:
- YOLOv8 (AI object detection).​
- Kaggle Datasets (Training images)​.
- Python Libraries (NumPy, OpenCV, TensorFlow, Pandas, matplotlib)​.
- GitHub (Version control)​.
- Other Libraries (Shutil, OS, sklearn, seaborn, YOLO).
- Installing necessary dependencies for image processing and augmentation.
- Divide the images into training, validation, and test sets using stratified sampling.
- Used LabelImg to label objects(caries) in the images.
- Setting up repositories and ensuring proper access for all team members.
- Added liscence GNU General Public License v3.0.  
- Using platforms for programming - Visual studio, Jupiter notebook.

# YOLOv8 Quickstart

See below for quickstart installation and usage examples. For comprehensive guidance on training, validation, prediction, and deployment, refer to the official full [Ultralytics Docs](https://docs.ultralytics.com/).

# YOLOv11 Quickstart

## How to Run the Model

### 1. Prepare the Dataset

```bash
python scripts/prepare_data.py
```

### 2. Train the Model

```bash
python scripts/train.py
```

Results will be saved across multiple training runs inside the `runs/detect/` directory.

---

## Install

Install the `ultralytics` package in a Python ≥ 3.8 environment with PyTorch ≥ 1.8:

```bash
pip install ultralytics
For alternative installation methods (Conda, Docker, source build), see the Ultralytics Quickstart Guide.
```

## Usage
Command Line Interface (CLI)
Run predictions, training, validation, and exports directly from your terminal:

# Predict with a pretrained YOLOv8n model
yolo predict model=yolov8n.pt For additional CLI examples, check the YOLO CLI Docs.

## Python API
Embed YOLOv8 into your Python projects with the same flexibility as the CLI:

```python
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Train on your dataset (e.g., coco8.yaml) for 50 epochs at 640px
model.train(data="coco8.yaml", epochs=50, imgsz=640, device="0")

# Validate on the validation split
model.val()

# Run inference on an image
results = model("path/to/image.jpg")
results[0].show()

# Export to ONNX
model.export(format="onnx")
```
# NumPy

here are the quick start installations steps and usage instructions for NumPy
https://numpy.org/install/ 

## Install

pip install numpy

## Usage
To perform the mathematical operations

# Open CV

here are the quick start installations steps and usage instructions for Open CV (https://pypi.org/project/opencv-python/#installation-and-usage )

## Install

pip install opencv

## Usage
Used for computer vision and processing

# Tensor Flow

here are the quick start installations steps and usage instructions for Tensor Flow(https://pypi.org/project/tensorflow/ )

## Install

pip install tensorflow

## usage
Used for high performance numerical computation.

# Pandas

here are the quick start installations steps and usage instructions for Pandas(https://pypi.org/project/pandas/#documentation )

## Install

pip install pandas

## Usage

Used for data analysis and manipulation.

# Matplotlib

here are the quick start installations steps and usage instructions for matplotlibs(https://matplotlib.org/stable/install/index.html )

## Install

pip install matplotlib 

## Usage

Used for data visualizations.

# sklearn

here are the quick start installations steps and usage instructions for sklearn(https://pypi.org/project/scikit-learn/ )

## Install

pip install scikit-learn

## Usage
To provide the comprehensive tool kit for machine learning

# seaborn

here are the quick start installations steps and usage instructions for seaborn(https://pypi.org/project/seaborn/ )

## Install

pip install seaborn

## Usage
Visualization library for statistical graphics plotting in python

# LabelImg 

here are the quick start installations steps and usage instructions for LabelImg(https://pypi.org/project/labelImg/ )

## Install

pip3 install labelImg
labelImg
labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE] 

## Usage
Used for annotaions of images


## Diagrams 

![image](https://github.com/user-attachments/assets/fb7a56a5-7277-462c-82ee-884cfe288e17)

![image](https://github.com/user-attachments/assets/73f8e3ee-b16a-47e1-9675-a028fbd27db2)

![image](https://github.com/user-attachments/assets/382a64d3-43bc-4c4c-87da-3ce47bb8d913)

![image](https://github.com/user-attachments/assets/0ba93639-ffb2-4948-83fc-573f6672015a)

![image](https://github.com/user-attachments/assets/7dfed78a-c6d3-48d4-8e2a-d55b7b83485a)

![image](https://github.com/user-attachments/assets/8cef7184-22ac-465d-b913-3128fcfaee90)

![image](https://github.com/user-attachments/assets/b60c21bb-e1f1-4dc7-9896-5acb0e351409)

![image](https://github.com/user-attachments/assets/166c5073-cc3e-4cdf-92c0-c7a9cd5d6c2c)

![image](https://github.com/user-attachments/assets/f22bc383-9529-4e46-9b95-c2f49fc189b5)

![image](https://github.com/user-attachments/assets/0a687003-9d9a-4932-b223-e83e173de9d5)


https://lucid.app/lucidchart/93a01739-b88e-43ed-9f9f-545a3ce7d15f/edit?viewport_loc=-1881%2C-1088%2C3846%2C1780%2C0_0&invitationId=inv_911cf3ad-47b1-4a91-9b0d-f85b0fd5f99a

# Progress Report (03/19/2025)
## Overview
This milestone focuses on the continued development of our project, including environment setup, project realization, research paper outline, diagrams, issue tracking, and presentation.

We have made significant progress, including identifying image clarity issues, implementing data augmentation techniques, graphing results, planning additional milestone steps, adding a license, debugging, processing augmentation code, stratified sampling based on the four subgroup of dataset, divided them into training, testing and validation sets, added labelling to the training images using labelimg, imported yolo 8 model, trained the model with training sets, and then tested the model with testing set of images. Our team has worked collaboratively to refine the dataset, ensuring consistency and clarity for future machine learning applications.

## Outcomes
* Identified image clarity issues: Evaluated the dataset and noted inconsistencies.
* Graphing methods: Visualized data through charts and graphs to analyze augmentation effects.
* Debugging code: Worked collaboratively to resolve issues in data processing scripts.
* Added a project license: Ensured proper documentation and licensing compliance.
* Collected and processed labeled image data from structured folders into a CSV file (dental_images.csv).
* Implemented a stratified sampling method for balancing the distribution of image classes across train, validation, and test sets.
* Splitted dataset into 80% training/validation and 20% test set, then splitted into 75% training and 25% validation from the train/val part.
* Developed a Python script to automate copying images into structured folders for each split (train/val/test) using the shutil module.
* Verified image distribution visually using bar plots for stratified columns across the different dataset splits.
* The images were annotated using Labellmg for accurate labeling and object detection preparation.
* Prepared code and data structure for modeling using tools like TensorFlow.
* Planned additional steps for Milestone 2: Defined upcoming tasks(data augmentation, including more sets of images for model accuracy).

## Hinderances
* Some augmentation techniques introduced distortions that required further refinement.

* Debugging the image processing code took longer than expected.

* Some team members faced minor access issues with repository permissions.

* Figuring out the stratified sampling was time taking.

* While Labelling images we had to do it manually which was time taking. 

## Ongoing Risks
| Risk | Status | Mitigation Strategy |
|------|--------|----------------------|
| Imbalances class distribustion| Mitigated | Implemented stratified sampling to balanced the data|
| Data processing bugs | Mitigated | Regular debugging sessions and peer reviews |
| Repository access issues | Addressed | Ensured all members have required permissions |
| Model performance | Pending | Data is organized and ready but, we can see the impact after the training of the model |
| Inconsistant folder or file name | Partially resolved | Resolving using bash commands |



# Research Paper Outline
## Abstract
Traditional dental examination methods involve manual inspections and X-ray imaging, which can be time-consuming, uncomfortable, and costly. AI-powered imaging technology presents an innovative alternative by leveraging machine learning techniques to analyze high-resolution intraoral images for cavity detection and plaque assessment. This research explores the effectiveness of AI-driven dental imaging, specifically employing the YOLOv8 object detection model to enhance diagnostic accuracy and patient comfort while minimizing reliance on traditional X-rays. Our approach involves dataset collection, augmentation, stratified sampling, image preprocessing, structured dataset pipelines and model training to optimize AI-assisted diagnostics. This paper will discuss the methodology, findings, and implications for improving efficiency in dental healthcare.

## Introduction
- Overview of traditional dental examination challenges
- Issues with patient anxiety, manual inspection errors, and high costs
- Limitations of X-ray-based diagnostics (scheduling, insurance coverage, radiation exposure)
- Rise in AI-powered imaging as an alternative solution
- Objective: Apply AI-driven imaging (YOLOv8) to improve early diagnosis, accuracy, and workflow in dental clinics

## Literature Review
- Overview of AI in medical image analysis (focus on dental applications) 
- Application of YOLO-based object detection in dentistry
- Previous work on automated detection of dental anomalies (plaque, cavities)
- Comparison of AI and traditional diagnostic methods
-	Ethical considerations in AI-based healthcare

## Methodology
## Dataset collection
- Collected dental image datasets from Kaggle and open-source platforms
- Applied stratified sampling to have balanced image representation across dental conditions

## Preprocessing and Augmentation
- Filtered low-resolution images to retain medium-to-high clarity samples.
- Applied image transformation methods like image resizing, cropping, rotation, normalization, and annotation.

## Model Development
- Trained YOLOv8 object detection model on annotated dental images.
- Fine-tuned hyperparameters based on validation feedback.
- Used modular Python scripts to automate folder creation and image reorganization using `shutil`, `os`, and `pandas`.

## Validation & Performance Analysis
- Evaluated model performance using precision, recall, and F1-score.
- Compared AI results with expert dentist annotations for reliability.

## Comparative Study
- Comparison between AI-based results and manual expert analysis.
- Cost-effectiveness and efficiency metrics.

### Results & Analysis:
- Graphical representation of model performance.
- Effectiveness of AI in identifying cavities and plaque.
- Success rate compared to traditional methods.
- Error rates and areas for improvement.
- Real-world feasibility of integrating AI into dental clinics.

## Challenges & Limitations
- Access to high-quality real-world dental images.
- Imbalanced class of images.
- Accuracy vs. false positives in cavity detection.
- Processing power and model training time.
- Dentist and patient acceptance of AI technology.

## Conclusion
- Summary of research findings.
- Implications for improving dental diagnostic procedures.
- AI as a transformative tool in oral healthcare.
- Recommendations for future development and real-world implementation.

## Diagrams & Figures
- **Workflow Diagram**: Overview of data collection, preprocessing, and model training.
- **AI Pipeline**: Step-by-step flow of YOLOv8 object detection for dental diagnostics.




















