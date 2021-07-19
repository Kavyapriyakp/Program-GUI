#________HEADER FILES_______

import tkinter
from tkinter import*
from tkvideo import tkvideo
from tkinter import ttk
from tkinter import filedialog
from _cffi_backend import callback
from PIL import ImageTk, Image
import cv2
from cv2 import *
import numpy as np
import sys
import time
import argparse
import imutils
from pathlib import Path

#________USER-DEFINED FUNCTIONS_______




kernel_d = np.ones((3,3), np.uint8)
kernel_e = np.ones((3,3), np.uint8)
kernel_gauss = (3,3)
dilate_times = 15                        #initializing_integer_variables
erode_times = 10                       #initializing_integer_variables
is_blur = True                          #initializing_boolean_variables
is_close = True                         #initializing_boolean_variables
is_draw_ct = False                      #initializing_boolean_variables
fac = 2                                 #initializing_integer_variables

#______INITALIZING THE GUI WINDOW_______

   
window=Tk()
window.configure(background="grey64");
window.title("BoSS")
window.resizable(0,0)
window.geometry('1300x680')

#______SETTING VARIBALES TO CHECK STATE OF BUTTON (CHECKED OR UNCHECKED)_______


clicked= StringVar()
chkValue1 = BooleanVar()
chkValue2 = BooleanVar()
current_value1 = IntVar()
current_value2 = IntVar()

def get_current_value1():
    return '{}'.format(current_value1.get())

def slider_changed1(event1):
    value_label.configure(text=get_current_value1())

slider_label1 = Label(window,text='k Value:',font=("Times New Roman",12),fg="black",bg="grey64").place(x=832,y=52)
slider1 = ttk.Scale(window, from_=0,to=10, orient='horizontal', command=slider_changed1, variable=current_value1).place(x=890,y=50)
value_label1 = ttk.Label(window, text=get_current_value1())
value_label1.place(x=995,y=52)


'''def get_current_value2():
    return '{}'.format(current_value2.get())

def slider_changed2(event2):
    value_label.configure(text=get_current_value2())

slider_label2 = Label(window,text='Parameter:',font=("Times New Roman",12),fg="black",bg="grey64").place(x=1058,y=52)
slider2 = ttk.Scale(window, from_=0,to=10, orient='horizontal', command=slider_changed2, variable=current_value2).place(x=1135,y=50)
value_label2 = ttk.Label(window, text=get_current_value2())
value_label2.place(x=1240,y=52)'''




#________CREATING BUTTONS_______


title = Label(window, text = "Border Surveillance System",font=("Times New Roman",18, 'bold'),fg="black",bg="grey64").place(x=495, y=10)
window.iconbitmap('DRDO.ico')

label_file_explorer = Label(window, text = "", fg = "blue")
label_file_explorer.grid(column = 1, row = 1)

#_______ADDING FUNCTIONALITES________

def browseFiles():
    source_file = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes =[('MP4 files', '.mp4'),('All Files', '.'),('ASF files', '.asf')],parent=window)
    label_file_explorer.configure(text=""+source_file)

    '''video_1 = Label(window)
    video_1.place(x=100,y=100)
    player1 = tkvideo(str(source_file), video_1, loop = 0, size = (500,500))
    player1.play()'''


    def drawRectangle(frame, minus_frame):
            if(is_blur):
                    minus_frame = GaussianBlur(minus_frame, kernel_gauss, 0)
            minus_Matrix = np.float32(minus_frame)	
            if(is_close):
                    for i in range(dilate_times):
                            minus_Matrix = dilate(minus_Matrix, kernel_d)
                    
                    for i in range(erode_times):
                            minus_Matrix = erode(minus_Matrix, kernel_e)
                    
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
            capture = VideoCapture(str(source_file));
            while(1):
                    (ret_old, old_frame) = capture.read()
                    
                    gray_oldframe = cvtColor(old_frame, COLOR_BGR2GRAY)
                    if(is_blur):
                            gray_oldframe = GaussianBlur(gray_oldframe, kernel_gauss, 0)
                    oldBlurMatrix = np.float32(gray_oldframe)
                    accumulateWeighted(gray_oldframe, oldBlurMatrix, 0.003)
                    while(True):
                            ret, frame = capture.read()
                            
                            gray_frame = cvtColor(frame, COLOR_BGR2GRAY)
                            if(is_blur):
                                    newBlur_frame = GaussianBlur(gray_frame, kernel_gauss, 0)
                            else:
                                    newBlur_frame = gray_frame
                            newBlurMatrix = np.float32(newBlur_frame)
                            minusMatrix = absdiff(newBlurMatrix, oldBlurMatrix)
                            ret, minus_frame = threshold(minusMatrix, 60, 255.0, THRESH_BINARY)
                            accumulateWeighted(newBlurMatrix,oldBlurMatrix,0.02)
                            imshow('Input', frame)
                            
                            drawRectangle(frame, minus_frame)
                            if cv2.waitKey(60) & 0xFF == ord('q'):
                                    break
                    capture.release() 
                    cv2.destroyAllWindows()

    objdetect()

    '''video_2 = Label(window)
    video_2.place(x=650,y=100)
    player2 = tkvideo(objdetect(), video_2, loop = 0, size = (500,500))
    player2.play()'''

C1=Button(window,text = "Browse",font=("Times New Roman",12, 'bold'),command=browseFiles).place(x=100,y=10)
C10=Checkbutton(window,text = "Input",font=("Times New Roman",12, 'bold'), background="grey64", foreground="black", var=chkValue1, state=DISABLED).place(x=140,y=50)
C2=Button(window,text="Live Input",font=("Times New Roman",12, 'bold'),state=DISABLED).place(x=300,y=10)
C20=Checkbutton(window,text = "Output",font=("Times New Roman",12, 'bold'), background="grey64", foreground="black", var=chkValue2, state=DISABLED).place(x=260,y=50)
C3=Button(window,text = "Object Detection",font=("Times New Roman",12, 'bold')).place(x=880,y=10)
C4=Button(window,text="Turbulence Mitigation",font=("Times New Roman",12, 'bold')).place(x=1090,y=10)




#______FOOTER OF THE GUI WINDOW_______



frame=LabelFrame(window,width=1300, height=50,fg="black",bg="aqua").place(x=0,y=630)
foot=Label(frame,text = "DIR/ECS/IRDE/PROC(BRR)/20-21/018",font=("Times New Roman",11),fg="black",bg="aqua").place(x=1010,y=645)
window.mainloop()

#_______END OF PROGRAM_______
