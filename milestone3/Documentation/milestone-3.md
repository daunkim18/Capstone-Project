# Realization Progress Report 
## Overview
Since Milestone 2, we significantly expanded our work on dental object detection by transitioning from YOLOv8 to YOLOv11, which was better suited for our dataset size and classification needs. We completed the model training, generated comprehensive evaluation metrics, finalized augmented datasets, and prepared a demo-ready pipeline for real-world testing. A complete GitHub repository, final documentation, and a polished slide presentation were also created.

## Outcomes
* Trained YOLOv11 model on 3,000 annotated intraoral images representing anterior caries, posterior caries, and healthy teeth.
* Improved model accuracy and generalization with manual annotation via Makesense.ai and strategic YOLO-compliant data formatting.
* Successfully visualized model outputs, producing image overlays with bounding boxes and class predictions.
* Implemented and tested training using augmented datasets (rotated, flipped, resized, cropped).
* We generated multiple visualizations to interpret model performance including:
  - Confusion Matrix (Raw & Normalized)
  - Precision-Recall and F1-Confidence Curves
  - Training loss curves across epochs Generated evaluation results:
mAP@0.5: 0.584
F1 Score: 0.63 at 0.43 confidence
Precision reached 1.0 at high thresholds
  
* Created clear visualizations: confusion matrix (raw/normalized), precision-recall curves, and performance loss metrics over epochs.
* Final presentation slides and script completed, covering model design, YOLOv11 advantages, and comparison with YOLOv8.
* Published code and documentation to GitHub with clean folder structure, training instructions, and README.md.

## Hinderances
- **Augmented image training was extremely slow**, with training time exceeding expectations even on CPU setups. We paused full training on the augmented set due to hardware/time limits.
- **Labeling was challenging**: labeling tools like LabelImg and Roboflow failed; final labels were successfully created with Makesense.ai and reuired expert dental knowledge for annotating dental caries or differentiating between anterior/posterior regions 
- **Balancing all 3 classes (anterior, posterior, healthy)** was difficult; healthy class remained underrepresented.
- **Small validation set** led to difficulty optimizing precision-recall tradeoff and caused some false positives. Some false positives remained in object detection; further training with augmented data needed. Balancing precision vs. recall in binary classification was challenging due to small validation set.
- **YOLOv8 underperformed**, especially with limited data. This led us to switch to YOLOv11, which offered better results.
- **Limited number of labeled images** reduced training variety.
- **Variability in image quality**, e.g., blurry, low-light, overexposed images affected model learning and consistency.
- **Inconsistent resolutions and aspect ratios**, for many images that requiring preprocessing and resizing that risked losing fine details.

**Conclusion:** This project demonstrates a strong technical effort, problem-solving initiative, and thoughtful progress toward a deployable AI diagnostic tool. The project is thoroughly documented and fully reproducible for evaluation or future development.


