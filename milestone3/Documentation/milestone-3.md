# Realization Progress Report 
## Overview
Since Milestone 2, we significantly expanded our work on dental object detection by transitioning from YOLOv8 to YOLOv11, which was better suited for our dataset size and classification needs. We completed the model training, generated comprehensive evaluation metrics, finalized augmented datasets, and prepared a demo-ready pipeline for real-world testing. A complete GitHub repository, final documentation, and a polished slide presentation were also created.

## Outcomes
* Trained YOLOv11 model on 3,000 annotated intraoral images representing anterior caries, posterior caries, and healthy teeth.
* Improved model accuracy and generalization with manual annotation via Makesense.ai and strategic YOLO-compliant data formatting.
* Successfully visualized model outputs, producing image overlays with bounding boxes and class predictions.
* Implemented and tested training using augmented datasets (rotated, flipped, resized, cropped).
* Generated evaluation results:
**mAP@0.5: 0.584
**F1 Score: 0.63 at 0.43 confidence
**Precision reached 1.0 at high thresholds
  
*Created clear visualizations: confusion matrix (raw/normalized), precision-recall curves, and performance loss metrics over epochs.
*Final presentation slides and script completed, covering model design, YOLOv11 advantages, and comparison with YOLOv8.
*Published code and documentation to GitHub with clean folder structure, training instructions, and README.md.

## Hinderances
* Limited number of labeled images reduced training variety.
* Some false positives remained in object detection; further training with augmented data needed.
* Balancing precision vs. recall in binary classification was challenging due to small validation set.
