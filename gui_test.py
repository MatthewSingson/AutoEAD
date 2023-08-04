# needed libraries to run
import tkinter as tk
from tkinter import messagebox
import numpy as np
from hmmlearn import hmm
from csv import reader
import cv2 as cv
import mediapipe as mp
import pandas as pd

# modularization
import mesh
import hmm_training
import J48_DT

def start_facial_extraction():
    filename = fileInputEntry.get()

    val = mesh.mesh_test_main(filename)
    if val.empty:
        messagebox.showerror('Error!',"File not found.")
    else:
        messagebox.showinfo('Complete!', "Feature Extraction Complete!")

def classify():
    filename = csvEntry.get()
    messagebox.showinfo('Classification', f"Entry : {filename}")

def rbClassification():
    pass

window = tk.Tk()

window.geometry("720x720")
window.title("AutoEAD")


fileInputFrame = tk.Frame(window, padx=4, pady=4)
fileInputFrame.columnconfigure(0, weight = 1)
fileInputFrame.columnconfigure(1, weight = 1)

#For video
fileInputText = tk.Label(fileInputFrame, height=1, font=('Arial, 12'), 
                         text="Video Clip Filename (no file extension):")
fileInputText.grid(row=0, column=0, sticky=tk.W + tk.E)

fileInputEntry = tk.Entry(fileInputFrame)
fileInputEntry.grid(row=0, column=1, sticky=tk.W + tk.E,)

fileInputButton = tk.Button(fileInputFrame, text="Start Feature Extraction", 
                            font=('Arial, 12'), command= start_facial_extraction)
fileInputButton.grid(row=1, column=1, sticky=tk.N + tk.E + tk.W)

fileInputFrame.pack(fill='x')

#For Classification
classificationFrame = tk.Frame(window, pady = 40)
classificationFrame.columnconfigure(0, weight = 1)
classificationFrame.columnconfigure(1, weight = 1)

eatCheck = tk.Radiobutton(classificationFrame, text= "Eating")
drinkCheck = tk.Radiobutton(classificationFrame, text = "Drinking")

eatCheck.grid(row = 0, column=1, sticky=tk.W + tk.E)
drinkCheck.grid(row = 1, column=1, sticky=tk.W + tk.E)

classificationBtn = tk.Button(classificationFrame, text = "Classify", 
                              command=classify, font = ('Arial', 12))

classificationBtn.grid(row = 2, column= 1, sticky=tk.W + tk.E)

csvLabel = tk.Label(classificationFrame, height = 1, 
                    font = ('Arial', 12), text="CSV filename (no file extension):")
csvLabel.grid(row = 0, column=0, sticky=tk.W + tk.E)

csvEntry = tk.Entry(classificationFrame)
csvEntry.grid(row=1, column=0, sticky=tk.W + tk.E)

classificationFrame.pack(fill='x')

#For HMM loading


window.mainloop()
