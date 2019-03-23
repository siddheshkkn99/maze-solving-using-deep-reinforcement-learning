# program to solve a maze  using deep reinforced learning technique

# ---------------------------------------------------------------------------------
# importing maze solver program

from mazeSolve import *
# ---------------------------------------------------------------------------------
# developing a sample maze - 7X7

# maze structure stored in a 2d numpy array with floating values from 0.0-1.0
# which will be plotted on tkinter 1.0 indicates free cell and 0.0 represents blocked cell
# the floating numbers also denote the color of the cell 1:white 0:black
# the rat is starting at cell (0,0) (top-left cell) and the cheese is placed at cell (9,9)
maze =  np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  0.,  1.,  0.],
    [ 0.,  0.,  0.,  1.,  1.,  1.,  0.],
    [ 1.,  1.,  1.,  1.,  0.,  0.,  1.],
    [ 1.,  0.,  0.,  0.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.]
])

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