# ---------------program to solve a maze  using deep reinforced learning technique---------------------------------

# ---------------------------------------------------------------------------------
# importing maze solver program
# importing libraries and functions
from __future__ import print_function
from IPython import get_ipython
import os, sys, time, datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
from mazeSolve import *
from mazeGenerator import *
# ---------------------------------------------------------------------------------
# developing a sample maze

# maze structure stored in a 2d numpy array with floating values from 0.0-1.0
# which will be plotted on tkinter 1.0 indicates free cell and 0.0 represents blocked cell
# the floating numbers also denote the color of the cell 1:white 0:black
# the rat is starting at cell (0,0) (top-left cell) and the cheese is placed at cell (9,9)

'''
sample
maze =  np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  0.,  1.,  0.],
    [ 0.,  0.,  0.,  1.,  1.,  1.,  0.],
    [ 1.,  1.,  1.,  1.,  0.,  0.,  1.],
    [ 1.,  0.,  0.,  0.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.]
])
'''
# generating sample 5X5 maze
maze=MazeGen(5,5).getMaze()

qmaze = Qmaze(maze)
show(qmaze)
# ----------------------------------------------------------------------------------

# feeding sample maze to the program
qmaze = Qmaze(maze)
show(qmaze)
plt.show()

# building a neural network model
model = build_model(maze)
qtrain(model, maze, epochs=1000, max_memory=8*maze.size, data_size=32)

# playing a game on the built model
play_game(model,qmaze,(0,0))
show(qmaze)
plt.show()
