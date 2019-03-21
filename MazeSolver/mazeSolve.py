# program to solve a maze  using deep reinforced learning technique
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
# ---------------------------------------------------------------------------------
# developing a sample maze

# maze structure stored in a 2d numpy array with floating values from 0.0-1.0
# which will be plotted on tkinter 1.0 indicates free cell and 0.0 represents blocked cell
# the floating numbers also denote the color of the cell 1:white 0:black
# the rat is starting at cell (0,0) (top-left cell) and the cheese is placed at cell (9,9)
maze = np.array([
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
LEFT    = 0
UP      = 1
RIGHT   = 2
DOWN    = 3
# setting up a dictionary for these variables:
actions_dict = \
{
    LEFT    : 'left',
    UP      : 'up',
    RIGHT   : 'right',
    DOWN    : 'down',
}
# variable to hold no. of actions possible by agent
num_actions = len(actions_dict)

# Exploration factor(epsilon): decides the probability that the agent will make a random move to discover new outcomes
epsilon = 0.1
# ---------------------------------------------------------------------------------------------

# Qmark class: describes the behaviour and characteristics of the maze cells included the rat and cheese
# initial rat position and target cell will always be top left cell = rat(0,0) and bottom right = cheese(lr,lc)
class Qmaze(object):

    # constructor function
    def __init__(self, _maze, rat=(0,0)):# initialises rat location to (0,0) which is its starting default location
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
        #resetting rats attributes for next try/iteration
        self.reset(rat)

    # reset function for rat
    def reset(self, rat):
        self.rat = rat
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = rat
        self.maze[row, col] = rat_mark
        #  set state as start state
        self.state = (row, col, 'start')
        # threshold reward below which the game is restarted
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()

    # function to update state of agent
    def update_state(self, action):
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state
        # mark visited cell
        if self.maze[rat_row, rat_col] > 0.0: self.visited.add((rat_row, rat_col))  
        # making a valid move
        valid_actions = self.valid_actions()
        # if there is no valid actions then the agent is blocked
        if not valid_actions: nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if   action == LEFT    :ncol -= 1
            elif action == UP      :nrow -= 1
            if   action == RIGHT   :ncol += 1
            elif action == DOWN    :nrow += 1
        # invalid action, no change in rat position
        else: mode = 'invalid'
        # setting new state
        self.state = (nrow, ncol, nmode)

    # function to set reward
    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        # reward for reaching target/cheese
        if rat_row == nrows - 1 and rat_col == ncols - 1    :return 1.0
        # penalty for getting blocked(restart)
        if mode == 'blocked'                                :return self.min_reward - 1
        # penalty for moving to a cell already visited
        if (rat_row, rat_col) in self.visited               :return -0.25
        # major penalty for making a invalid move
        if mode == 'invalid'                                :return -0.75
        # minor penalty for making a move to an unvisited cell
        if mode == 'valid'                                  :return -0.04

    # function to set environment based on agent actions
    def act(self, action):
        #update actions
        self.update_state(action)
        # set reward
        reward = self.get_reward()
        self.total_reward += reward
        # set game status
        status = self.game_status()
        # set environment
        envstate = self.observe()
        return envstate, reward, status

    # function to store the tkinter canvas
    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    # function to draw the environment on the tkinter canvas
    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0: canvas[r, c] = 1.0
        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = rat_mark
        return canvas

    # function to set status of game(win/loos/not over)
    def game_status(self):
        # restart game if total reward goes below threshold
        if self.total_reward < self.min_reward: return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        # the game is won or stopped when agent reaches target
        if rat_row == nrows - 1 and rat_col == ncols - 1: return 'win'
        return 'not_over'

    # function to check if a move is valid or not
    def valid_actions(self, cell=None):
        if cell is None :row, col, mode = self.state
        else            :row, col = cell
        # list of actions possible
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        # removing actions not possible because it would go beyond the maze
        if   row == 0           : actions.remove(1)
        elif row == nrows - 1   : actions.remove(3)
        if   col == 0           : actions.remove(0)
        elif col == ncols - 1   : actions.remove(2)
        # removing actions that cant be made because of blocked cells around agent
        if row > 0         and self.maze[row - 1, col] == 0.0   : actions.remove(1)
        if row < nrows - 1 and self.maze[row + 1, col] == 0.0   : actions.remove(3)
        if col > 0         and self.maze[row, col - 1] == 0.0   : actions.remove(0)
        if col < ncols - 1 and self.maze[row, col + 1] == 0.0   : actions.remove(2)
        return actions

