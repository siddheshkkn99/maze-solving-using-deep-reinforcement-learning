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

# ----------------------------------------------------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------------------------------------------------

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
# ----------------------------------------------------------------------------------------------------------------------
# Experience Class:This is the class in which we collect our game episodes (or game experiences) within a memory list

# It takes the following parameters: a neural network, maximum memory ,and discount factor

class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d maze cells info, including rat cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory: del self.memory[0]

    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]   # envstate 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))
            if game_over: targets[i, action] = reward
            # reward + gamma * max_a' Q(s', a')
            else: targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets
# ----------------------------------------------------------------------------------------------------------------------
# Q-Training: algorithm for training our neural network model to solve the maze

# It takes the parameters: no of training epochs, max memory and data size
def qtrain(model, maze, **opt):
    global epsilon
    n_epoch = opt.get('n_epoch', 15000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()

    # If you want to continue training from a previous model,
    # just supply the h5 file name to weights_file option
    if weights_file:
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)

    # Construct environment/game from numpy array: maze (see above)
    qmaze = Qmaze(maze)

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory)

    win_history = []  # history of win/lose game
    n_free_cells = len(qmaze.free_cells)
    hsize = qmaze.maze.size // 2  # history window size
    win_rate = 0.0
    imctr = 1

    for epoch in range(n_epoch):
        loss = 0.0
        rat_cell = random.choice(qmaze.free_cells)
        qmaze.reset(rat_cell)
        game_over = False

        # get initial envstate (1d flattened canvas)
        envstate = qmaze.observe()

        n_episodes = 0
        while not game_over:
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            prev_envstate = envstate
            # Get next action
            if np.random.rand() < epsilon: action = random.choice(valid_actions)
            else: action = np.argmax(experience.predict(prev_envstate))

            # Apply action, get reward and new envstate
            envstate, reward, game_status = qmaze.act(action)
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
            else:
                game_over = False

            # Store episode (experience)
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1

            # Train neural network model
            inputs, targets = experience.get_data(data_size=data_size)
            h = model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0,
            )
            loss = model.evaluate(inputs, targets, verbose=0)

        if len(win_history) > hsize: win_rate = sum(win_history[-hsize:]) / hsize

        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch - 1, loss, n_episodes, sum(win_history), win_rate, t))
        # we simply check if training has exhausted all free cells and if in all
        # cases the agent won
        if win_rate > 0.9: epsilon = 0.05
        if sum(win_history[-hsize:]) == hsize and completion_check(model, qmaze):
            print("Reached 100%% win rate at epoch: %d" % (epoch,))
            break

    # Save trained model weights and architecture, this will be used by the visualization code
    h5file = name + ".h5"
    json_file = name + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)
    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    return seconds
# ----------------------------------------------
# Building a neural network model:

def build_model(maze, lr=0.001):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model
# -------------------------------------------------------------------------------------------
# Utility Functions:

# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)

# function to plot maze current state on canvas
def show(qmaze):
    # matplotlib commands to setup the environment
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # copying maze to tkinter canvas for display
    canvas = np.copy(qmaze.maze)
    # marking visited cells with dark gray color(50%)
    for row,col in qmaze.visited: canvas[row,col] = 0.6
    rat_row, rat_col, _ = qmaze.state
    # marking rat cell with very dark gray(30%)
    canvas[rat_row, rat_col] = 0.3
    # marking cheese cell with light gray(90%)
    canvas[nrows-1, ncols-1] = 0.9
    # plotting canvas
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img

# function to play the game, it takes 3 arguments: model,maze and the rat position
def play_game(model, qmaze, rat_cell):
    # reset rat state for new game/iteration
    qmaze.reset(rat_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])
        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if   game_status == 'win' : return True
        elif game_status == 'lose': return False

#  simulate all possible games and check if our model wins them all( not optimal for large mazes)
def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell): return False
        if not play_game(model, qmaze, cell): return False
    return True



