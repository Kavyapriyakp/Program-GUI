<h1 align="center"><a name="section_name">Program-GUI</a></h1>

<div align="justify">
A GUI model of a software prototype with latest features developed to aid surveillance systems' monitoring.
</div>

## Project Title
<div align="justify">
The developed GUI corresponds to the project titled "Reuqirement of real-time compensation for turbulence and detecting moving objects", developed in 2020-21.
</div>

## About
<div align="justify">
The objective of the project is to develop an algorithm for detecting and rectifying the  atmospheric turbulence, Spatio-temporal blur, hence improving the quality of input video  using the electro-optical systems and finding all moving objects including vehicles and  humans. The aim is to develop an approach that combines multi-frame image reconstruction  with Gaussian Mixture Modeling (GMM) based moving object detection and General  Adversarial Networks (GAN) model to de-blur input videos. 
</div>

## Build Status

<img src="https://img.shields.io/badge/build-passing-brightgreen"/> <img src="https://img.shields.io/badge/code-latest-orange"/> <img src="https://img.shields.io/badge/langugage-python-blue"/>


## Intended Use

<img src="https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white"/> <img src="https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white"/>


## Screenshot / GIF / Video

## Hardware and Software Requirements

<div align="justify">
The following frameworks were utilized in the building of this code.
</div>


<img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white"/> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/> <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>


<div align="justify">
The following tools were utilized in the building of this code.
</div>


<img src="https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white"/> <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white"/>


<div align="justify">
The code was originally created on
</div>


<img src="https://img.shields.io/badge/NVIDIA-GTX1650-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>


## Features
<div align="justify">
The following features are integrated in the given code. Each of these features are explained in detail in it's respective repository.
</div>

1. Background Modification
   1. Blur
   2. Subtraction
2. Detection
   1. Moving Objects
   2. Object Edges
   3. Intruder Alert
   4. Object Cut
3. Classification & Segmentation
   1. Instance
   2. Semantic
4. Quality Metrics
   1. Image
   2. Video
   3. IOU
   4. Jaccard

## Code Example

The following table mentions all the user-defined functions and it's corresponding description.

| Function                  | Description                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------ |
| `cameradetect`          | detects if any surveillance device is connected, else throws error                                     |
| `edgedetect`            | detects and marks boundaries of objects within images                                                  |
| `backgroundsubtraction` | extracs foreground and background of image                                                             |
| `bgblur`                | applies a low-pass filter to blur outlier pixles                                                       |
| `bgcolour`              | sets a colour to outlier pixels                                                                        |
| `inst_seg`             | detects distinct instances of objects of interest in the image                                         |
| `sem_seg`              | detecs objects of different class of interest in the image                                             |
| `drawRectangle`         | - encloses the moving object within a bounding<br>- takes 2 consecutive images / frames as arguments |
| `objdetect`             | - detects moving objects<br>- draws bouding box by calling `drawRectangle`|


```Python
import time
import sys
```


## Project Structure

├── app
│   ├── css
│   │   ├── **/*.css
│   ├── favicon.ico
│   ├── images
│   ├── index.html
│   ├── js
│   │   ├── **/*.js
│   └── partials/template
├── dist (or build)
├── node_modules
├── bower_components (if using bower)
├── test
├── Gruntfile.js/gulpfile.js
├── README.md
├── package.json
├── bower.json (if using bower)
└── .gitignore


## Installation & Dependencies / Development Setup

<div align="justify">
This program is originally intended to work on GPUs.
</div>

## API Reference

## User Guide

## Work under Progress

## Credit / Contributors & Owners

## Licence

## Feedback

[Go to Top](#section_name)
