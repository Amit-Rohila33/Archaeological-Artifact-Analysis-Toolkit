# Archaeological Artifact Analysis Toolkit

This toolkit leverages deep learning models like Vision Transformers (ViTs) and YOLOv8 to analyze archaeological artifacts. It segments, classifies, and predicts the cultural and historical context of artifacts using image and text data.

## Installation

Clone the repository and install dependencies:

```bash
git clone <repo_url>
cd Archaeological-Artifact-Analysis-Toolkit
pip install -r requirements.txt


### **Running the Project**
1. Preprocess the images by running the preprocessing script.
2. Use `segment.py` to run semantic segmentation and identify artifact types.
3. Use `detect.py` to classify objects using YOLOv8.
4. Extract metadata and predict context using `extract_metadata.py`.