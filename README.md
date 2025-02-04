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

## Methodology
This study will build upon previous research but integrate the latest advancements in object detection, specifically utilizing YOLOv8 for multi-granularity tooth analysis and automated dental plaque detection. The research model will follow a data-driven approach, leveraging high-resolution intraoral images to train and validate AI models for tooth classification, cavity detection, and plaque quantification. The methodology will include data collection from diverse dental case studies, preprocessing and augmentation of images, and iterative model training and validation to ensure robustness and accuracy. By employing an updated YOLOv8 framework, the study will refine the detection accuracy beyond prior research models, ensuring enhanced performance in real-time clinical settings. The outcomes of this model will be validated through comparative analysis with traditional dental examination methods, ensuring the feasibility and reliability of AI-assisted diagnostics. 

Key Steps & Methodology:

1. Data Collection & Preprocessing
   Dataset: The oral images (caries vs. non-caries) will be labeled from a pre-existing dataset. Images will be preprocessed by resizing for uniformity. The images will be converted to NumPy arrays for processing and normalizing (rescaling pixel values) to [0,1] for     better convergence.
2. Data Augmentation
   To enhance model generalization and prevent overfitting, ImageDataGenerator will be used. The dataset diversity will be increased by applying rotation, width shift, height    shift, zooming, and flipping.
3. Model Developement
4. Model Training & Evaluation
5. Model Prediction & Testing
6. Result

## Resources Needed

| Resource                                      | Dr. Hale Needed? | Investigating Team Member | Description |
|-----------------------------------------------|------------------|---------------------------|-------------|
| Google Colab                                  | Yes               | Sravani                      | Cloud-based platform for training and testing the YOLOv8 model. |
| YOLOv8 (Ultralytics)                          | No               | Sravani                      | Pre-trained object detection model for dental image analysis. |
| Kaggle Dental Imaging Dataset                 | No/Yes               | Sravani                      | Publicly available dataset for training and validating the model. |
| Local Dental School Imaging Data              | No              | Daun                      | Requesting real-world dental images for model enhancement. |
| Python (NumPy, OpenCV, Pandas, TensorFlow)    | Yes               | Reshmi                      | Required libraries for data processing, model training, and evaluation. |
| GitHub                                        | No               | Daun                      | Version control for project collaboration and documentation. |
| Computational Resources (Colab Pro/Local GPU) | Yes              | Reshmi                      | Possible need for extended training time or additional GPU resources. |


# Milestone 2
## Environment setup
Regardless of the type of project you are working on, you will need some work environments. You should determine tooling needs, document their usage/installation/configuration and include that information in your project repostiory.
provide a clear pathway to setting up and configuring needed tools to begin working on your project. For maker-based projects this might be software dependencies and coding libraries or establishing some build files (in something like Docker). For breaker-based projects it might be creating VMs or docker containers to containerize the target of investigations.
Google Colab (Cloud-based model training)​

- YOLOv8 (AI object detection)​
- Kaggle Datasets (Training images)​
- Local Dental School Data (Pending approval)​
- Python Libraries (NumPy, OpenCV, TensorFlow, Pandas)​
- GitHub (Version control)​
- Other Libraries (Shutil, OS, JSON, YOLO)

## Diagrams 

![image](https://github.com/user-attachments/assets/fb7a56a5-7277-462c-82ee-884cfe288e17)

![image](https://github.com/user-attachments/assets/73f8e3ee-b16a-47e1-9675-a028fbd27db2)



