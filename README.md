
# **Low-Light Image Stitching Using Classical Techniques**

A project to create seamless panoramas by stitching multiple images together, focusing on low-light conditions using classical computer vision methods learned in class.

---

## **Table of Contents**
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## **Introduction**
Image stitching is the process of combining overlapping images to create a wide-field panorama. Low-light conditions make this task challenging due to reduced contrast and poor feature detectability.

This project addresses these issues with:
- Preprocessing to enhance low-light images.
- Feature detection and matching using classical methods.
- Homography estimation and blending to create a panorama.

---

## **Features**
- Preprocess low-light images using:
  - Brightness mapping and histogram equalization.
  - Noise reduction with Gaussian filtering.
- Extract and match features using:
  - Classical detectors: Sobel, Harris corner detection.
  - Matching with RANSAC for robust alignment.
- Seamless blending of stitched images.

---

## **Installation**
### **Prerequisites**
- Python 3.8 or higher
- Libraries: OpenCV, NumPy, Matplotlib

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/QuodFinis/low-light-image-stitching.git
   cd low-light-stitching
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**
### **1. Preprocess Low-Light Images**
Enhance image brightness and contrast:
```bash
python preprocess.py --input images/low_light/ --output images/preprocessed/
```

### **2. Run Image Stitching**
Stitch images using the classical pipeline:
```bash
python stitch.py --input images/preprocessed/ --output results/panorama.jpg
```

### **3. Evaluate the Results**
Evaluate the quality of the stitched panorama:
```bash
python evaluate.py --ground_truth images/ground_truth/ --predicted results/panorama.jpg
```

---

## **Methodology**
1. **Preprocessing**:
   - Brightness and contrast enhancement using histogram equalization.
   - Noise reduction with Gaussian filtering.
2. **Feature Detection and Matching**:
   - Detect features using Harris corner detection or Sobel edge detection.
   - Match features using simple nearest-neighbor matching and refine with RANSAC.
3. **Homography Estimation**:
   - Use matched features to compute the homography matrix.
   - Align images using this matrix.
4. **Blending**:
   - Combine images using simple feathering to reduce visible seams.

---

## **Dataset**
### **Image Stitching Datasets**:
- **UAV Image Stitching Dataset**: [Link](https://github.com/droneslab/uav-image-stitching-dataset).
- **Microsoft ICE Sample Datasets**: [Link](https://www.microsoft.com/en-us/research/product/computational-photography-applications/image-composite-editor/).

---

## **Evaluation**
- **Quantitative Metrics**:
  - Feature Matching Accuracy.
  - Reprojection Error.
- **Qualitative Metrics**:
  - Visual seamlessness and alignment quality.

---

## **Results**
### **Baseline Pipeline**
- Feature Matching Accuracy: X%
- Reprojection Error: X
- Visual Observations: (Insert stitched image screenshots)

---

## **Contributing**
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit changes:
   ```bash
   git commit -m "Add feature"
   ```
4. Push the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
