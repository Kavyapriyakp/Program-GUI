<h1 align="center"><a name="section_name">Program-GUI</a></h1>

<div align="justify">
A GUI model of a software prototype with latest features developed to aid surveillance systems' monitoring.
</div>

## Project Title
<div align="justify">
The developed GUI corresponds to the project titled "Requirement of real-time compensation for turbulence and detecting moving objects", developed in 2020-21.
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
   3. Colour
2. Detection
   1. Moving Objects
   2. Object Edges
   3. Intruder Alert
3. Classification & Segmentation
   1. Instance
   2. Semantic
4. Quality Metrics
   1. Image
   2. Video
   3. IOU
   4. Jaccard




## Project Structure

├── Data Models<br>
│   ├── List of data models.txt<br>
├── Input<br>
│   ├── 01<br>
│   ├── 02<br>
│   ├── 03<br>
├── Logos<br>
│   ├── DRDO.png<br>
├── Output<br>
│   ├── 01<br>
│   ├── 02<br>
│   ├── 03<br>
├── Program<br>
│   ├── GUI.py<br>
├── Requirements<br>
│   ├── requirements.txt<br>
└── README.md<br>

## User Guide


## Installation & Dependencies / Development Setup


<div align="justify">

The GUI code supports *tensorflow*'s version (2.0 - 2.4.1). Install *tensorflow* using:

```Python
pip3 install tensorflow
```

If you have have a PC enabled GPU, install *tensorflow--gpu*'s version that is compatible with the cuda installed on your pc:


```Python
pip3 install tensorflow--gpu
```


Install the latest version of *Pixellib* with:

```Python
pip3 install pixellib --upgrade
```
</div>

<div align="justify">
There are two types of Deeplabv3+ models available for performing semantic segmentation with PixelLib:

1. Deeplabv3+ model with xception as network backbone trained on Ade20k dataset, a dataset with 150 classes of objects.
2. Deeplabv3+ model with xception as network backbone trained on Pascalvoc dataset, a dataset with 20 classes of objects. 

Instance segmentation is implemented with PixelLib by using Mask R-CNN model trained on coco dataset. The latest version of PixelLib supports custom training of object segmentation models using pretrained coco model.  Deeplab and mask r-ccn models are available in the above mentioned data models directory and can be downloaded from [here](https://drive.google.com/drive/folders/1jtSFQN3W6_tkF5QVUYto5slp_wvntIQO?usp=sharing).

</div>

## Code Example

The following table mentions all the user-defined functions and it's corresponding description.

| Method                  | Description                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------ |
| `cameradetect()`          | detects if any surveillance device is connected, else throws error                                     |
| `edgedetect()`            | detects and marks boundaries of objects within images                                                  |
| `backgroundsubtraction()` | extracs foreground and background of image                                                             |
| `bgblur()`                | applies a low-pass filter to blur outlier pixles                                                       |
| `bgcolour()`              | sets a colour to outlier pixels                                                                        |
| `inst_seg()`             | detects distinct instances of objects of interest in the image                                         |
| `sem_seg()`              | detecs objects of different class of interest in the image                                             |
| `drawRectangle(frame, minus_frame)`         | - encloses the moving object within a bounding<br>- takes 2 consecutive images / frames as arguments |
| `objdetect()`             | - detects moving objects<br>- draws bouding box by calling `drawRectangle` method|


<details>
<summary>Reading video input and extracting frames</summary>

```Python
cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('<file_path>')
```
<div align="justify">

`cap` is the object on `VideoCapture()` method to capture a video. It accepts either the device index or the name of a video file. 

The `cap.read()` returns a boolean value.  It will return True, if the frame is read correctly.
</div>

```Python
while(1): 
        ret, frame = cap.read()
```
<div align="justify">
This code initiates an infinite loop (to be broken later by a break statement), where we have ret and frame being defined as the cap.read(). Basically, ret is a boolean regarding whether or not there was a return at all, at the frame is each frame that is returned. If there is no frame, you wont get an error, you will get None.
</div>

```Python
cv2.imshow('Input',frame)
```

```Python
k = cv2.waitKey(5) & 0xFF
if k == ord("q"): 
    break
```

```Python
cap.release() 
cv2.destroyAllWindows()
```

</details>


<details>
<summary>Using in-built Python functions</summary>

```Python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
```

```Python
lower_red = np.array([30,150,50]) 
upper_red = np.array([255,255,180]) 
mask = cv2.inRange(hsv, lower_red, upper_red) 
res = cv2.bitwise_and(frame,frame, mask= mask)  
```

```Python
edges = cv2.Canny(frame,100,200)
```

```Python
fgbg = cv2.createBackgroundSubtractorMOG2()
```

```Python
fgmask = fgbg.apply(frame)
```

```Python
kernel_d = np.ones((3,3), np.uint8)
kernel_e = np.ones((3,3), np.uint8)
```
```Python
if(is_blur):
	minus_frame = GaussianBlur(minus_frame, kernel_gauss, 0)
minus_Matrix = np.float32(minus_frame)	
```

```Python
minus_Matrix = np.clip(minus_Matrix, 0, 255)
	minus_Matrix = np.array(minus_Matrix, np.uint8)
```

```Python
contours, hierarchy = findContours(minus_Matrix.copy(), RETR_TREE, CHAIN_APPROX_SIMPLE)
```

```Python
(x, y, w, h) = boundingRect(c)	
rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
```		

</details>


<details>
<summary>Creating the GUI</summary>


```Python
window=Tk()
window.configure(background="grey64");
window.title("Border Surveillance System")
window.resizable(0,0)
window.geometry('850x600')
```	

```Python
clicked  = StringVar()
chkValue = BooleanVar()
```	

```Python
window.mainloop()
```	

</details>

## API Reference

## Work under Progress

## Credit / Contributors & Owners

## References

1. Bonlime, Keras implementation of Deeplab v3+ with pretrained weights  https://github.com/bonlime/keras-deeplab-v3-plus
2. Mask R-CNN code made compatible with tensorflow 2.0, https://github.com/tomgross/Mask_RCNN/tree/tensorflow-2.0
3. TensorFlow DeepLab Model Zoo https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md

[Go to Top](#section_name)
