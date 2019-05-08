import tkinter
from tkinter import Label
from tkinter import *
from clasifier import analyze_picture_linear
from clasifier import start_webcam_linear
import pickle
import PIL
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
import numpy as np
from facifier import analyze_picture
from facifier import start_webcam

window = tkinter.Tk()
window.geometry('1250x270')
window.title("Identificarea emoțiilor persoanelor")


l1 = Label(window, text="Emotion Recognition", font = ("Arial Bold", 30))
l1.grid(column = 4, row = 0)
l2 = Label(window, text = "Choose your classifier", font = ("Arial", 20))
l2.grid(column = 4, row = 5,  pady = 10)

def select_image():

        global panelA 
        panelA = None
        loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
        path = filedialog.askopenfilename(initialdir = "../data/sample",title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png"),("all files","*.*")))
        window_name = "Emotion Recognition"
        if len(path)>0:
            image = cv2.imread(path)
            height1 = np.size(image, 0)
            width1 = np.size(image, 1)
            image = analyze_picture_linear(loaded_model, path, window_size=(750, 250), window_name=window_name)
            #open("results_image.txt")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)   
            if panelA is None:
                panelA = Label(image = image, width=width1, height=height1)
                panelA.image = image
            else:
                panelA.grid_forget()
                panelA.configure(image=image)
                panelA.image = image

        else:
            print("Could not do it!")

def create_window_image():
    window3 = tkinter.Tk()
    window3.geometry("200x50")
    window3.title("Linear SVM")
    panelA = None
    bt = Button(window3, text = "Select an image", font = ("Arial", 15), command = select_image)
    bt.grid(column = 5, row = 0, padx = "10", pady = "10")

    window3.mainloop

def create_window_video():
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    window_name = "Emotion Recognition"
    start_webcam_linear(loaded_model, window_size=(750, 250), window_name=window_name, update_time=15)


def select_image_fisher():

        global panelB 
        panelB = None
        fisher_face_emotion = cv2.face.FisherFaceRecognizer_create()
        fisher_face_emotion.read('models/emotion_classifier_model.xml')
        window_name = "Emotion Recognition"
        path2 = filedialog.askopenfilename(initialdir = "../data/sample",title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png"),("all files","*.*")))
        if len(path2)>0:
            image2 = cv2.imread(path2)
            height1 = np.size(image2, 0)
            width1 = np.size(image2, 1)
            image2 = analyze_picture(loaded_model, path, window_size=(750, 250), window_name=window_name)
            image2 = Image.fromarray(image2)
            image2 = ImageTk.PhotoImage(image2)   
            if panelB is None:
                panelB = Label(image = image2, width=width1, height=height1)
                panelB.image = image2
            else:
                panelB.grid_forget()
                panelB.configure(image=image2)
                panelB.image = image2

        else:
            print("Could not do it!")


def create_window_video_fisher():
    fisher_face_emotion = cv2.face.FisherFaceRecognizer_create()
    fisher_face_emotion.read('models/emotion_classifier_model.xml')
    window_name = "Emotion Recognition"
    start_webcam(fisher_face_emotion, window_size=(750, 250), window_name=window_name, update_time=15)

def create_window_image_fisher():
    window4 = tkinter.Tk()
    window4.geometry("200x50")
    window4.title("FisherFace classifier")
    panelB = None
    bt_B = Button(window4, text = "Select an image", font = ("Arial", 15), command = select_image)
    bt_B.grid(column = 5, row = 0, padx = "10", pady = "10")
    window4.mainloop


b_fisher_face = Label(window, text = "Fisher Face Classifier", font = ("Arial", 15))
b_fisher_face.grid(column = 3, row = 7)

b_linear = Label(window, text = "Linear SVM", font = ("Arial", 15))
b_linear.grid(column = 6, row = 7)

radio_fisher_face_images = Button(window, text = "Static images", font = ("Arial", 15), command = create_window_image_fisher)
radio_video_fisher_face = Button(window, text = "Video", font = ("Arial", 15), command = create_window_video_fisher)
radio_fisher_face_images.grid(column = 2, row = 8, padx = "10")
radio_video_fisher_face.grid(column = 4, row = 8, sticky = "wn", padx = 10)

radio_linear_static_images = Button(window, text = "Static images", font = ("Arial", 15), command = create_window_image)
radio_video = Button(window, text = "Video", font = ("Arial", 15), command = create_window_video)
radio_linear_static_images.grid(column = 5, row = 8, padx = 10)
radio_video.grid(column = 7, row = 8, sticky = "wn", padx = 10)

window.mainloop()
