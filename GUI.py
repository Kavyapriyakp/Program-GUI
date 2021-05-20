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
import sys
import pixellib
from pixellib.instance import instance_segmentation
from pixellib.tune_bg import alter_bg
from pixellib.semantic import semantic_segmentation
import numpy as np
import argparse
import imutils

#______________________________________________________________ALL FUNCTIONS___________________________________________________________________

def cameradetect():
    cap = cv2.VideoCapture('rtsp://admin:Password@123@192.168.1.64') 
    while(1): 
        ret, frame = cap.read() 
        cv2.imshow('Input',frame)   
        k = cv2.waitKey(5) & 0xFF
        if k == ord("q"): 
            break
    cap.release() 
    cv2.destroyAllWindows()


def edgedetect():
    cap = cv2.VideoCapture('rtsp://admin:Password@123@192.168.1.64') 
    while(1): 
        ret, frame = cap.read() 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
        lower_red = np.array([30,150,50]) 
        upper_red = np.array([255,255,180]) 
        mask = cv2.inRange(hsv, lower_red, upper_red) 
        res = cv2.bitwise_and(frame,frame, mask= mask) 
        edges = cv2.Canny(frame,100,200) 
        cv2.imshow('Input',frame) 
        cv2.imshow('Edges',edges)  
        k = cv2.waitKey(5) & 0xFF
        if k == ord("q"): 
            break
    cap.release() 
    cv2.destroyAllWindows()


def backgroundsubtraction():
    cap = cv2.VideoCapture('rtsp://admin:Password@123@192.168.1.64') 
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

change_bg = alter_bg(model_type = "pb")
change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
def bgblur():
    capture = cv2.VideoCapture('rtsp://admin:Password@123@192.168.1.64')
    change_bg.blur_camera(capture, frames_per_second=10,extreme = True, show_frames = True, frame_name = "frame", output_video_name="bgblur_out.mp4")


def bgcolour():
    capture = cv2.VideoCapture('rtsp://admin:Password@123@192.168.1.64')
    change_bg.color_camera(capture, frames_per_second=10,colors = (0, 128, 0), show_frames = True, frame_name = "frame", output_video_name="output_video.mp4")

    
def inst_seg():
    capture = cv2.VideoCapture('rtsp://admin:Password@123@192.168.1.64')
    segment_video = instance_segmentation(infer_speed = "rapid")
    segment_video.load_model("mask_rcnn_coco.h5")
    segment_video.process_camera(capture, frames_per_second= 10, show_bboxes = True, show_frames= True,frame_name= "frame", output_video_name="inst_seg_out.mp4")

def sem_seg():
    capture = cv2.VideoCapture('rtsp://admin:Password@123@192.168.1.64')
    segment_video = semantic_segmentation()
    segment_video.load_ade20k_model("deeplabv3_xception65_ade20k.h5")
    segment_video.process_camera_ade20k(capture, overlay=True, frames_per_second= 10, output_video_name="sem_seg_out.mp4", show_frames= True,frame_name= "frame")

    
kernel_d = np.ones((3,3), np.uint8)
kernel_e = np.ones((3,3), np.uint8)
kernel_gauss = (3,3)
dilate_times = 13
erode_times = 5
is_blur = True
is_close = True
is_draw_ct = False
fac = 2
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
	capture = VideoCapture('rtsp://admin:Password@123@192.168.1.64');
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

def action_recog():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model",help="path to trained human activity recognition model")
    ap.add_argument("-c", "--classes",help="path to class labels file")
    ap.add_argument("-i", "--input", type=str, default="",
            help="optional path to video file")
    args = vars(ap.parse_args())
    CLASSES = open(args["classes"] if args["classes"] else "action_recognition_kinetics.txt").read().strip().split("\n")
    SAMPLE_DURATION = 16
    SAMPLE_SIZE = 112
    print("[INFO] loading human activity recognition model...")
    net = cv2.dnn.readNet(args["model"] if args["model"] else "resnet-34_kinetics.onnx")
    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
    while True:
            frames = []
            for i in range(0, SAMPLE_DURATION):
                    (grabbed, frame) = vs.read()
                    if not grabbed:
                            print("[INFO] no frame read from stream - exiting")
                            sys.exit(0)
                    frame = imutils.resize(frame, width=400)
                    frames.append(frame)
            blob = cv2.dnn.blobFromImages(frames, 1.0,
                    (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
                    swapRB=True, crop=True)
            blob = np.transpose(blob, (1, 0, 2, 3))
            blob = np.expand_dims(blob, axis=0)
            net.setInput(blob)
            outputs = net.forward()
            label = CLASSES[np.argmax(outputs)]
            for frame in frames:
                    cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
                    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
                    cv2.imshow("Activity Recognition", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                            break

#______________________________________________________________MAIN PROGRAM___________________________________________________

   
window=Tk()
window.configure(background="grey64");
window.title("Border Surveillance System")
window.resizable(0,0)
window.geometry('850x600')


clicked= StringVar()
chkValue1 = BooleanVar()
chkValue2 = BooleanVar()
chkValue3 = BooleanVar()
chkValue4 = BooleanVar()
chkValue5 = BooleanVar()
chkValue6 = BooleanVar()
chkValue7 = BooleanVar()
chkValue8 = BooleanVar()


title = Label(window, text = "CARS-SRMIST",font=("Times New Roman",20, 'bold'),fg="black",bg="grey64").place(x= 350, y=10)
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

#_______________________________________________________________ALL BUTTONS__________________________________________________________________________

C1=Checkbutton(window,text = "Object Focus",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black",var=chkValue1,command=bgblur).place(x=400,y=120)
C2=Checkbutton(window,text = "Background Colour",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black",var=chkValue2,command=bgcolour).place(x=400,y=150)
C3=Checkbutton(window,text="Background Subtraction",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black",var=chkValue3,command=backgroundsubtraction).place(x=400,y=180)
C4=Checkbutton(window,text="Edge Detection",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black",var=chkValue4,command=edgedetect).place(x=400,y=210)
C5=Checkbutton(window,text="Object Detection",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black",var=chkValue5,command=objdetect).place(x=400,y=240)
C6=Checkbutton(window,text="Instance Segmentation",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black",var=chkValue6,command=inst_seg).place(x=600,y=120)
C7=Checkbutton(window,text="Semantic Segmentation",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black",var=chkValue7,command=sem_seg).place(x=600,y=150)
C8=Checkbutton(window,text="Action Detection",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black",var=chkValue8,command=action_recog).place(x=600,y=180)
C9=Checkbutton(window,text="-",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black").place(x=600,y=210)
C10=Checkbutton(window,text="-",font=("Times New Roman",12, 'bold'),background="grey64",foreground="black").place(x=600,y=240)
b4=Button(window,text = "Execute",font=("Times New Roman",12, 'bold')).place(x=550, y=300)
parameter=Button(window,text = "Parameters",font=("Times New Roman",14, 'bold'),state=DISABLED).place(x=380, y=395)


photo = PhotoImage(file="./DRDO.png")
frame=LabelFrame(window,width=850, height=150,fg="black",bg="aqua").place(x=0,y=450)
foot=Label(frame,text = "Developed For",font=("Times New Roman",13, 'bold'),fg="black",bg="aqua").place(x=30,y=460)
imgLabel = Label(frame,image=photo,bg="aqua").place(x=40, y=490)
textLabel =Label(frame,text = "  Instruments Research & \n Development Establishment",font=("Times New Roman",11, 'bold'),fg="black",bg="aqua").place(x=135,y=510)
window.mainloop()



