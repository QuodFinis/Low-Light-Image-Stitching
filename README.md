# Low-Light-Image-Stitching


A project exploring classical computer vision techniques to improve image stitching performance in low-light conditions. The aim is to develop a preprocessing-based pipeline to enhance feature detection, matching, and blending for seamless image stitching.

Table of Contents

	•	Introduction
	•	Features
	•	Installation
	•	Usage
	•	Methodology
	•	Dataset
	•	Evaluation
	•	Results
	•	Contributing
	•	License

Introduction

Image stitching is the process of aligning and blending multiple images to create a panoramic view. This project focuses on improving image stitching in low-light conditions, where traditional methods struggle due to poor feature detection and matching.

Key improvements include:
	•	Preprocessing for low-light image enhancement.
	•	Robust feature matching and homography estimation.
	•	Improved blending techniques to reduce ghosting and seams.

Features

	•	Low-light preprocessing:
	•	Histogram equalization (global and CLAHE).
	•	Denoising and edge enhancement.
	•	Classical feature detection and matching:
	•	Algorithms: SIFT, SURF, ORB.
	•	Outlier rejection using Lowe’s ratio test and RANSAC.
	•	Advanced image blending techniques:
	•	Feathering and multi-band blending.
	•	Quantitative and qualitative evaluation metrics.

Installation

Prerequisites

	•	Python 3.8 or higher
	•	OpenCV
	•	NumPy
	•	Matplotlib
	•	Scikit-image

Setup

	1.	Clone the repository:

git clone https://github.com/yourusername/low-light-image-stitching.git
cd low-light-image-stitching


	2.	Install dependencies:

pip install -r requirements.txt

Usage

1. Preprocess Low-Light Images

Enhance low-light images using histogram equalization or denoising:

python preprocess.py --input images/low_light/ --output images/preprocessed/

2. Run the Stitching Pipeline

Stitch images using the classical pipeline:

python stitch.py --input images/preprocessed/ --output results/panorama.jpg

3. Evaluate Performance

Evaluate the stitching quality:

python evaluate.py --ground_truth images/ground_truth/ --predicted results/panorama.jpg

Methodology

	1.	Feature Detection and Matching:
	•	Extract features using SIFT, SURF, or ORB.
	•	Match features with a nearest-neighbor approach and refine with Lowe’s ratio test.
	2.	Homography Estimation:
	•	Use RANSAC to compute the homography matrix and align images.
	3.	Blending:
	•	Combine images using blending techniques like feathering or multi-band blending.
	4.	Preprocessing:
	•	Enhance low-light images to improve feature detection.

Dataset

Sources:

	•	ExDark Dataset
	•	Custom low-light images captured using a smartphone or camera.
	•	Augmented datasets created from COCO or other sources by simulating low-light conditions.

Evaluation

	•	Quantitative Metrics:
	•	Feature Matching Accuracy.
	•	Reprojection Error.
	•	Structural Similarity Index (SSIM).
	•	Qualitative Metrics:
	•	Visual seamlessness and alignment quality.

Results

Baseline Pipeline

	•	Feature Matching Accuracy: X%
	•	Reprojection Error: X
	•	Visual Observations: (Add screenshots of stitched images)

Improved Pipeline

	•	Feature Matching Accuracy: Y%
	•	Reprojection Error: Y
	•	Visual Observations: (Add side-by-side comparison of baseline vs improved results)

Contributing

Contributions are welcome! Please follow these steps:
	1.	Fork the repository.
	2.	Create a feature branch: git checkout -b feature-name.
	3.	Commit changes: git commit -m 'Add some feature'.
	4.	Push to the branch: git push origin feature-name.
	5.	Open a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.
