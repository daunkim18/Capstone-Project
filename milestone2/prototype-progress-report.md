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
* Prepared code and data structure for modeling using tools like TensorFlow.
* Planned additional steps for Milestone 2: Defined upcoming tasks(data augmentation, including more sets of images for model accuracy).

## Hinderances
* Some augmentation techniques introduced distortions that required further refinement.

* Debugging the image processing code took longer than expected.

* Some team members faced minor access issues with repository permissions.

## Ongoing Risks
| Risk | Status | Mitigation Strategy |
|------|--------|----------------------|
| Image clarity loss due to augmentation | Ongoing | Adjust augmentation parameters and test different techniques |
| Data processing bugs | Mitigated | Regular debugging sessions and peer reviews |
| Repository access issues | Addressed | Ensured all members have required permissions |
