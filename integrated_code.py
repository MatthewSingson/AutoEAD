# needed libraries to run
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

hmm_eat_filename = "eating_train_HMM"
hmm_drink_filename = "drinking_train_HMM"
vid_filename = "r02 - 02.2e"
model_filename_drinking = ""
model_filename_eating = "eating_gaussian_model.pkl"

#mesh code call
mesh.mesh_test_main(vid_filename)
x = J48_DT.decision_tree_eating(vid_filename)
print(f"Classified as: {x}")

#hmm code call
#hmm_training.hmm_eat_train(hmm_eat_filename, model_filename_eating)
#hmm_training.hmm_drink_train(hmm_drink_filename, "M")


