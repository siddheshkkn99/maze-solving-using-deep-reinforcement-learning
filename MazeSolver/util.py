# Utility Functions:

import matplotlib.pyplot as plt
import numpy as np


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