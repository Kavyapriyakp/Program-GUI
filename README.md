<h1 align="center"><a name="section_name">Program-GUI</a></h1>

<div align="justify">
A GUI model of a software prototype with latest features developed to aid surveillance systems' monitoring.
</div>

## Project Title
<div align="justify">

</div>

## About
<div align="justify">
 
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
├── Input<br>
│   ├── Frames<br>
│   ├── Videos<br>
├── Logos<br>
├── Output<br>
│   ├── Frames<br>
│   ├── Videos<br>
├── Program<br>
├── Requirements<br>
└── README.md<br>

## User Guide

<div align="justify">

Start off by downloading the required models from the *Data Models* directory.  Move the downloaded models to the parent directory.  Sample input and output are depcited in the *Input* and *Output* directories respectively.  These are provided for reference, one can expect a similar output for the given input.  The *Program* directory contains the Python code for the said GUI.  Move the file to the parent directory.  The *Requirements* directory contains a list of necessary packages.  Now execute the *GUI.py* file and voila!

</div>

## Installation of Dependencies / Development Setup

### Pre-requisites


<div align="justify">

The code is developed with Python ver.3.8 and `pip` ver.21.0.1 in Windows OS and the same is tested on Ubuntu OS too. The necessary packages and frameworks can be installed from the *Requirements* directory.  

```Python
pip install -r requirements.txt
``` 

However, one can follow the below mentioned steps in case of any errors.


Firstly, check the version of Python on your system using:

```Python
python --version
``` 

If you wish to change / upgrade the version or install Python afresh, visit https://www.python.org/downloads/. 

`pip` is a package-management system written in Python used to install and manage software packages. It connects to an online repository of public packages, called the Python Package Index. pip can also be configured to connect to other package repositories.  One can check `pip` version using:

```Python
pip --version
```

If you wish to install `pip` afresh:

```Python
python3 -m pip install --upgrade pip
```

or

```Python
sudo apt install python3-pip
```

Installing the necessary packages and depencies is a pre-requisite.  The setup itself varies according to the OS, though the code is really the same.  Yet, the GUI is builded with different libraries in runtime, hence it results in differrent appearances of the same GUI accroding to OSs.



<details>
<summary>Windows OS</summary>

---

The `tkinter` package (“Tk interface”) is the standard Python interface to the Tk GUI toolkit. The `Tk interface` is located in a binary module named `_tkinter`. It is usually a shared library (or DLL), but might in some cases be statically linked with the Python interpreter.  The `cffi` module is used to invoke `callback` methods inside the program.

```Python
pip install tk
python3 -m pip install cffi
```

`Pillow` is a Python Imaging Library (`PIL`), which adds support for opening, manipulating, and saving images. The current version identifies and reads a large number of formats.  It supports wide variety of images such as “jpeg”, “png”, “bmp”, “gif”, “ppm”, “tiff”.

```Python
python3 -m pip install --upgrade Pillow
```

`OpenCV` is a huge open-source library for computer vision, machine learning, and image processing. `OpenCV` supports a wide variety of programming languages like Python, C++, Java, etc. It can process images and videos to identify objects, faces, and so on. The library has more than 2500 optimized algorithms, which includes a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms.


```Python
pip install opencv-python
```

The GUI code supports `tensorflow`'s version (2.0 - 2.4.1). Install `tensorflow` and the latest version of `Pixellib` with:

```Python
pip3 install tensorflow
pip3 install pixellib --upgrade
```

If you have have a PC enabled GPU, install *tensorflow--gpu*'s version that is compatible with the cuda installed on your pc:


```Python
pip3 install tensorflow--gpu
```

`NumPy` is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.  By reading the image as a `NumPy` array ndarray, various image processing can be performed using NumPy functions.


```Python
pip3 install numpy
```


`Imutils` are a series of convenience functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, and displaying Matplotlib images easier with `OpenCV` in Python.


```Python
pip3 install imutils
```


</details>


<details>
<summary>Ubuntu OS</summary>

---

The `tkinter` package (“Tk interface”) is the standard Python interface to the Tk GUI toolkit. The `Tk interface` is located in a binary module named `_tkinter`. It is usually a shared library (or DLL), but might in some cases be statically linked with the Python interpreter.  The `cffi` module is used to invoke `callback` methods inside the program.

```Python
apt-get install python-tk 
sudo apt-get install python-setuptools
sudo apt-get install -y python-cffi
```

`Pillow` is a Python Imaging Library (`PIL`), which adds support for opening, manipulating, and saving images. The current version identifies and reads a large number of formats.  It supports wide variety of images such as “jpeg”, “png”, “bmp”, “gif”, “ppm”, “tiff”.

```Python
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade Pillow
```


`OpenCV` is a huge open-source library for computer vision, machine learning, and image processing. `OpenCV` supports a wide variety of programming languages like Python, C++, Java, etc. It can process images and videos to identify objects, faces, and so on. The library has more than 2500 optimized algorithms, which includes a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms.


```Python
sudo apt-get install python3-opencv
```
 
The GUI code supports `tensorflow`'s version (2.0 - 2.4.1). Install `tensorflow` and the latest version of `Pixellib` with:

```Python
pip3 install tensorflow
pip3 install pixellib --upgrade
```

If you have have a PC enabled GPU, install *tensorflow--gpu*'s version that is compatible with the cuda installed on your pc:


```Python
pip3 install tensorflow--gpu
```

`NumPy` is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.  By reading the image as a `NumPy` array ndarray, various image processing can be performed using NumPy functions.


```Python
pip3 install numpy
```


`Imutils` are a series of convenience functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, and displaying Matplotlib images easier with `OpenCV` in Python.


```Python
pip3 install imutils
```



</details>



</div>


### Using the Data Models


<div align="justify">

Instance segmentation is implemented with `PixelLib` by using Mask R-CNN model trained on coco dataset. The latest version of PixelLib supports custom training of object segmentation models using pretrained coco model.  Deeplab and Mask R-CNN models are available in the above mentioned *Data Models* directory. 

There are two types of Deeplabv3+ models available for performing semantic segmentation with PixelLib:

1. Deeplabv3+ model with xception as network backbone trained on Ade20k dataset, a dataset with 150 classes of objects.
2. Deeplabv3+ model with xception as network backbone trained on Pascalvoc dataset, a dataset with 20 classes of objects. 


The said data models can also be downloaded from [here](https://drive.google.com/drive/folders/1jtSFQN3W6_tkF5QVUYto5slp_wvntIQO?usp=sharing).

</div>

## Code Example

<details>
<summary>Reading video input and extracting frames</summary>

---


```Python
cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('<file_path>')
```
<div align="justify">

`cap` is the object on `VideoCapture()` method to capture a video. It accepts either the device index or the name of a video file as argument. 
The `cap.read()` returns a boolean value.  It will return True, if the frame is read correctly.
</div>


```Python
while(1): 
        ret, frame = cap.read()
```

<div align="justify">

This code initiates an infinite loop (to be broken later by a `break` statement), where `ret` and `frame` are being defined by the `cap.read()` method. `ret` is a boolean regarding whether or not there was a return at all.  Error is thrown if there is no frame.


</div>


```Python
cv2.imshow('Input',frame)
```

<div align="justify">

`cv2.imshow(window_name, image)` method is used to display an image in a window. The window automatically fits to the image size.
`window_name` argument is a string representing the name of the window in which image to be displayed. 
`image` argument is the image that is to be displayed.


```Python
k = cv2.waitKey(5) & 0xFF
if k == ord("q"): 
    break
```

The `waitKey(int)` method returns -1 when no input is made. As soon the event occurs it returns a 32-bit integer. It takes an integer argument, that is time in milliseconds (0 – wait indefinitely).
 `0xFF` represents a binary `11111111`, a 8 bit binary.  Since 8 bits are required to represent a character, `AND` operation is performed on `waitKey(int)` to `0xFF`. As a result, an integer is obtained below 255.
`ord(char)` returns the ASCII value of the character which would be pf a value not more than 255. Hence by comparing the integer to the `ord(char)` value, a check of whether a key is pressed for the event to break the loop is done.


```Python
cap.release() 
cv2.destroyAllWindows()
```

The `cap.release()` method closes video file or releases the connnect video capturing device.
The `cv2.destroyAllWindows()` method destroys all the windows that has been created. To
destroy any specific window, `cv2.destroyWindow()` is used where the exact window name is passed as argument.


</div>

</details>


<details>
<summary>Using in-built Python functions</summary>

---

<div align="justify">


```Python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
```
The `cv2. cvtColor(input_image, flag)` method is used for colour conversion where flag determines the type of conversion. For BGR to Gray conversion, the flag `cv2.COLOR_BGR2GRAY` is used. Similarly for BGR to HSV, the flag `cv2.COLOR_BGR2HSV` is used.


```Python
lower_red = np.array([30,150,50]) 
upper_red = np.array([255,255,180])
```

The `np.array([x,y,z])` method represents a grid of grid of values of the same type, and is indexed by a tuple of nonnegative integers. It takes a single argument as a tuple containing the number of dimensions (rank of the array) and the shape of an array (integers giving the size of the array along each dimension).


```Python
edges = cv2.Canny(frame,100,200)
```

The `cv.Canny(image, threshold1, threshold2, apertureSize, L2gradient)` method of the `cv2` library is used to detect edges in an image.  It can take upto 5 arguments: input image of n-dimensional array, high threshold value of intensity gradient, low threshold value of intensity gradient, order of matrix, a boolean type variable; of which the latter two are optional.


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
The `boundingRect()` method of OpenCV is used to draw an approximate rectangle around the image. It is used mainly to highlight the region of interest after obtaining contours from an image.  It takes the contours of the image as argument.
The `rectangle()` method is used to draw a rectangle and takes many arguments like image on which it has to be drawn, parameters for dimensionsleft, top, right, bottom, width, height, line thickness and so on.


```Python

change_bg.blur_camera(capture, frames_per_second=10,extreme = True, show_frames = True, frame_name = "frame", output_video_name="bgblur_out.mp4")


change_bg.color_camera(capture, frames_per_second=10,colors = (0, 128, 0), show_frames = True, frame_name = "frame", output_video_name="output_video.mp4")

    
segment_video = instance_segmentation(infer_speed = "rapid")            #setting_the_speed
segment_video.load_model("mask_rcnn_coco.h5")                           #loading_the_datamodel
segment_video.process_camera(capture, frames_per_second= 10, show_bboxes = True, show_frames= True,frame_name= "frame", output_video_name="inst_seg_out.mp4")

segment_video.load_ade20k_model("deeplabv3_xception65_ade20k.h5")       #loading_the_datamodel
segment_video.process_camera_ade20k(capture, overlay=True, frames_per_second= 10, output_video_name="sem_seg_out.mp4", show_frames= True,frame_name= "frame")
```	

The above mentioned code snipped is used to blur the background, assign colour to it, perform instance and segmentation on the given input.Futher explanation on the code snippet can be found [here](https://github.com/Kavyapriyakp/PixelLib/tree/master/Tutorials).


<div>
</details>


<details>
<summary>Creating the GUI</summary>

---

<div align="justify">


```Python
window=Tk()
window.configure(background="grey64");
window.title("Border Surveillance System")
window.resizable(0,0)
window.geometry('850x600')
```	

`Tk()` method is used to create and initilize the GUI window. 
`configure()` method is used to set a `background` (colour)  and a number of other parameters to the window.
`title()` method is used to set a name to the window.  Its takes a string as an argument.
`resizable()` method is used to the window to change it's size according to the user's need or prohibit resizing of the window.
`geometry()` method is used to set dimensions to the window. It takes the width and height as argurments. 

```Python
clicked  = StringVar()
chkValue = BooleanVar()
```	

Some widgets (like text entry widgets, radio buttons and so on) can be connected directly to application variables by using special options: variable, textvariable, onvalue, offvalue, and value. This connection works both ways: if the variable changes for any reason, the widget it's connected to will be updated to reflect the new value. 
`StringVar()` - Holds a string; with default value "" or NULL
`BooleanVar()` - holds a boolean, returns 0 for False and 1 for True


```Python
window.mainloop()
```	
The `mainloop()` method is used when the application is ready to run. It is an infinite loop used to run the application, wait for an event to occur and process the event as long as the window is not closed.

Many other `tkinter` widgets such as `CheckButton`, `Button`, `Combobox` and various other methods suuch as `Label()`, `.place()`, `.set()`, `LabelFrame()` are used in this code.  More details about `tkinter` can be found in [GeeksforGeeks](https://www.geeksforgeeks.org/python-gui-tkinter/).

</div>

</details>

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


## Work under Progress

### API Reference

## Credit / Contributors & Owners

## References

1. Bonlime, Keras implementation of Deeplab v3+ with pretrained weights  https://github.com/bonlime/keras-deeplab-v3-plus
2. Mask R-CNN code made compatible with tensorflow 2.0, https://github.com/tomgross/Mask_RCNN/tree/tensorflow-2.0
3. TensorFlow DeepLab Model Zoo https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
4. Pixellib - Python Library for Real-time Segmentation https://github.com/ayoolaolafenwa/PixelLib 
5. Bo Yang, Mingyue Tang, Shaohui Chen, Gang Wang, Yan Tan & Bijun Li “A vehicle tracking algorithm combining detector and tracker” EURASIP Journal on Image and Video Processing volume 2020, Article number: 17 (2020)


[Go to Top](#section_name)
