# program to deploy an agent(rat) in a maze who will find the target cell(cheese)
# this programs uses reinforced learning technique
# ---------------------------------------------------------------------------------

#prerequites

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

# setting the backend of matplotlib to the 'inline' backend
'exec(%matplotlib inline)'

# ---------------------------------------------------------------------------------

# developing a sample maze

# maze structure stored in a 2d numpy array with floating values from 0.0-1.0
# which will be plotted on tkinter 1.0 indicates free cell and 0.0 represents blocked cell
# the floating numbers also denote the color of the cell 1:white 0:black
# the rat is starting at cell (0,0) (top-left cell) and the cheese is placed at cell (9,9)
maze = np.array\
([
    [1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  1.],
    [1.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  1.],
    [1.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    [1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
    [1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.]
])

# ---------------------------------------------------------------------------------------

# setting up maze traversal constants

# cells visited by agent marked gray (0.8)
visited_mark = 0.8

# cells where agent(rat) is currently present is marked dark gray (0.5)
rat_mark = 0.5

# We have exactly 4 actions the agent can perform which are moving  in 4 direction across the maze
# left,right,up,down which we must encode as integers 0-3
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

# setting up a dictionary for these variables:
actions_dict = \
{
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

# variable to hold no. of actions possible by agent
num_actions = len(actions_dict)

# Exploration factor(epsilon): decides the probability that the agent will make a random move to discover new outcomes
epsilon = 0.1

# ---------------------------------------------------------------------------------------------

# Qmark class: describes the behaviour and characteristics of the maze cells included the rat and cheese

class Qmaze(object):

    # constructor function
    def _init_(self, _maze, rat=(0,0)):# initialises rat location to (0,0) which is its starting default location

        # the maze nparray is added as another nparray
        self._maze = np.array(_maze)

        # no.of rows and column will be calculated using shape function of numpy library
        nrows, ncols = self._maze.shape

        # target cell (cheese) is set as the right corner cell by default
        self.target = (nrows-1, ncols-1)

        #free_cells is a list containing coordinates of all free cells and removing target cell from it
        # because target cell is a special case and cannot be considered as a ordinary free cell
        self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if(self._maze[r, c] == 1.0)]
        self.free_cells.remove(self.target)

        # raising exception in cases where rat goes to a blocked cell or outside maze OR if target is blocked
        if(self._maze[self.target]) == 0.0: raise Exception("Target Cell/Cheese cannot be blocked")
        if(not (rat in self.free_cells)): raise Exception("Agent/Rat not present in a free cell")

        #resetting rats attributes for new run
        self.reset(rat)

    # reset function of rat for the new run
    def reset(self, rat):
        
        #set rat attributes for current run
        self.rat = rat
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = rat
        
        # marks color corresponding to rat(dark gray) on cell
        self.maze[row, col] = rat_mark
        
        # set the state of rat's current location
        self.state = (row, col, 'start')
        
        # min reward threshold which will loose the game if actual reward goes below this value
        self.min_reward = -0.5 * self.maze.size
        
        # resetting rat reward to 0
        self.total_reward = 0 
        
        # initialise visited cell variableas a set() data structure
        self.visited = set()
