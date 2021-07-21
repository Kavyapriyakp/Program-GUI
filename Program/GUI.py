#______________________________________________________________HEADER FILES___________________________________________________________________

import tkinter
from tkinter import*
from tkinter import ttk
from _cffi_backend import callback
from PIL import ImageTk, Image
import cv2
from cv2 import *
import numpy as np
import sys
import time
import pixellib
from pixellib.instance import instance_segmentation
from pixellib.tune_bg import alter_bg
from pixellib.semantic import semantic_segmentation
import argparse
import imutils

#______________________________________________________________USER-DEFINED FUNCTIONS___________________________________________________________________

def cameradetect():
    cap = cv2.VideoCapture(0) 
    while(1): 
        ret, frame = cap.read() 
        cv2.imshow('Input',frame)   
        k = cv2.waitKey(5) & 0xFF
        if k == ord("q"): 
            break
    cap.release() 
    cv2.destroyAllWindows()


def edgedetect():
    cap = cv2.VideoCapture(0) 
    while(1): 
        ret, frame = cap.read() 
        edges = cv2.Canny(frame,100,200) 
        cv2.imshow('Input',frame) 
        cv2.imshow('Edges',edges)  
        k = cv2.waitKey(5) & 0xFF
        if k == ord("q"): 
            break
    cap.release() 
    cv2.destroyAllWindows()


def backgroundsubtraction():
    cap = cv2.VideoCapture(0) 
    fgbg = cv2.createBackgroundSubtractorMOG2() 
        
    while(1): 
        ret, frame = cap.read() 
        fgmask = fgbg.apply(frame)  
        cv2.imshow('Input',frame) 
        cv2.imshow('fgmask',fgmask)         
        k = cv2.waitKey(60) & 0xff
        if k == ord("q"): 
            break  
            
    cap.release() 
    cv2.destroyAllWindows()


change_bg = alter_bg(model_type = "pb")                     #loading_data_model
change_bg.load_pascalvoc_model("xception_pascalvoc.pb")     #loading_data_model


def bgblur():
    capture = cv2.VideoCapture(0)
    change_bg.blur_camera(capture, frames_per_second=10,extreme = True, show_frames = True, frame_name = "frame", output_video_name="bgblur_out.mp4")


def bgcolor():
    capture = cv2.VideoCapture(0)
    change_bg.color_camera(capture, frames_per_second=10,colors = (0, 128, 0), show_frames = True, frame_name = "frame", output_video_name="output_video.mp4")

    
def inst_seg():
    capture = cv2.VideoCapture(0)
    segment_video = instance_segmentation(infer_speed = "rapid")            #setting_the_speed
    segment_video.load_model("mask_rcnn_coco.h5")                           #loading_the_datamodel
    segment_video.process_camera(capture, frames_per_second= 10, show_bboxes = True, show_frames= True,frame_name= "frame", output_video_name="inst_seg_out.mp4")

def sem_seg():
    capture = cv2.VideoCapture(0)
    segment_video = semantic_segmentation()
    segment_video.load_ade20k_model("deeplabv3_xception65_ade20k.h5")       #loading_the_datamodel
    segment_video.process_camera_ade20k(capture, overlay=True, frames_per_second= 10, output_video_name="sem_seg_out.mp4", show_frames= True,frame_name= "frame")


kernel_d = np.ones((3,3), np.uint8)
kernel_e = np.ones((3,3), np.uint8)
kernel_gauss = (3,3)
dilate_times = 13                       #initializing_integer_variables
erode_times = 5                         #initializing_integer_variables
is_blur = True                          #initializing_boolean_variables
is_close = True                         #initializing_boolean_variables
is_draw_ct = False                      #initializing_boolean_variables
fac = 2                                 #initializing_integer_variables
def drawRectangle(frame, minus_frame):
	if(is_blur):
		minus_frame = GaussianBlur(minus_frame, kernel_gauss, 0)
	minus_Matrix = np.float32(minus_frame)	
	if(is_close):
		for i in range(dilate_times):
			minus_Matrix = dilate(minus_Matrix, kernel_d)
		imshow('dilate', minus_Matrix)
		for i in range(erode_times):
			minus_Matrix = erode(minus_Matrix, kernel_e)
		imshow('erode', minus_Matrix)
	minus_Matrix = np.clip(minus_Matrix, 0, 255)
	minus_Matrix = np.array(minus_Matrix, np.uint8)
	contours, hierarchy = findContours(minus_Matrix.copy(), RETR_TREE, CHAIN_APPROX_SIMPLE)
	for c in contours:
		(x, y, w, h) = boundingRect(c)	
		rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		if( is_draw_ct ):
			drawContours(frame, contours, -1, (0, 255, 255), 2)
	imshow('result', frame)

def objdetect():
	capture = VideoCapture(0);
	width = (int)( capture.get( CAP_PROP_FRAME_WIDTH )/fac )
	length = (int)( capture.get( CAP_PROP_FRAME_HEIGHT )/fac )
	while(1):
		(ret_old, old_frame) = capture.read()
		old_frame = resize( old_frame, ( width,length ),interpolation = INTER_CUBIC )
		gray_oldframe = cvtColor(old_frame, COLOR_BGR2GRAY)
		if(is_blur):
			gray_oldframe = GaussianBlur(gray_oldframe, kernel_gauss, 0)
		oldBlurMatrix = np.float32(gray_oldframe)
		accumulateWeighted(gray_oldframe, oldBlurMatrix, 0.003)
		while(True):
			ret, frame = capture.read()	
			frame = resize( frame, ( width,length ),interpolation = INTER_CUBIC )
			gray_frame = cvtColor(frame, COLOR_BGR2GRAY)
			if(is_blur):
				newBlur_frame = GaussianBlur(gray_frame, kernel_gauss, 0)
			else:
				newBlur_frame = gray_frame
			newBlurMatrix = np.float32(newBlur_frame)
			minusMatrix = absdiff(newBlurMatrix, oldBlurMatrix)
			ret, minus_frame = threshold(minusMatrix, 60, 255.0, THRESH_BINARY)
			accumulateWeighted(newBlurMatrix,oldBlurMatrix,0.02)
			imshow("difference", minus_frame)
			drawRectangle(frame, minus_frame)
			if cv2.waitKey(60) & 0xFF == ord('q'):
				break
		capture.release() 
		cv2.destroyAllWindows()


#________________________________________________________INITALIZING THE GUI WINDOW___________________________________________________________________

   
window=Tk()
window.configure(background="grey64");
window.title("Surveillance System")
window.resizable(0,0)
window.geometry('850x500')

#____________________________________________SETTING VARIBALES TO CHECK STATE OF BUTTON (CHECKED OR UNCHECKED)___________________________________________________________________


clicked= StringVar()
chkValue1 = BooleanVar()
chkValue2 = BooleanVar()
chkValue3 = BooleanVar()
chkValue4 = BooleanVar()
chkValue5 = BooleanVar()
chkValue6 = BooleanVar()
chkValue7 = BooleanVar()
chkValue8 = BooleanVar()


#____________________________________________________________CREATING BUTTONS_______________________________________________________________


title = Label(window, text = "Survillance System Monitor",font=("Times New Roman",20, 'bold'),fg="black",bg="grey64").place(x= 252, y=10)
t1=Label(window,text = "Input",font=("Times New Roman",16, 'bold'),fg="black",bg="grey64").place(x=150,y=80)
b1=Button(window,text = "Browse",font=("Times New Roman",12, 'bold'),state=DISABLED).place(x=50, y=130)
b2=Button(window,text = "Live Video",font=("Times New Roman",12, 'bold'),command=cameradetect).place(x=200, y=130)
device=StringVar()
device=ttk.Combobox(window,width=30,height=6, state='readonly',textvariable=device)
device.place(x=50,y=220)
device.set("Choose a Surveillance Device :")
device['values']=('Device 1','Device 2','Device 3')
device.current()
b3=Button(window,text = "Select",font=("Times New Roman",9, 'bold')).place(x=255, y=218)  
t2= Label(window,text = "Modules",font=("Times New Roman",16, 'bold'),fg="black",bg="grey64").place(x=550,y=80)


#___________________________________________________________ADDING FUNCTIONALITES__________________________________________________________________________


C1=Checkbutton(window,text = "Object Focus",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black",var=chkValue1,command=bgblur).place(x=400,y=120)
C2=Checkbutton(window,text = "Background Colour",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black",var=chkValue2,command=bgcolor).place(x=400,y=150)
C3=Checkbutton(window,text="Background Subtraction",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black",var=chkValue3,command=backgroundsubtraction).place(x=400,y=180)
C4=Checkbutton(window,text="Edge Detection",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black",var=chkValue4,command=edgedetect).place(x=400,y=210)
C6=Checkbutton(window,text="Instance Segmentation",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black",var=chkValue6,command=inst_seg).place(x=600,y=120)
C7=Checkbutton(window,text="Semantic Segmentation",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black",var=chkValue7,command=sem_seg).place(x=600,y=150)
C8=Checkbutton(window,text="Intruder Alert",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black",var=chkValue8,command=edgedetect).place(x=600,y=180)
C9=Checkbutton(window,text="Object Detection",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black",var=chkValue5,command=objdetect).place(x=600,y=210)
b4=Button(window,text = "Execute",font=("Times New Roman",12, 'bold'),state=DISABLED).place(x=550, y=300)
parameter=Button(window,text = "Parameters",font=("Times New Roman",14, 'bold'),state=DISABLED).place(x=380, y=395)


#________________________________________________________FOOTER OF THE GUI WINDOW___________________________________________________________________



frame=LabelFrame(window,width=850, height=50,fg="black",bg="aqua").place(x=0,y=450)
foot=Label(frame,text = "Developed on & Supports - Python 3.8.x",font=("Times New Roman",11),fg="black",bg="aqua").place(x=600,y=465)
window.mainloop()

#_____________________________________________________________END OF PROGRAM___________________________________________________________________


