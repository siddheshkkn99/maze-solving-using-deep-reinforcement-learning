# ---------------program to solve a maze  using deep reinforced learning technique---------------------------------

# importing maze generator and solver programs
from mazeSolve import *
from mazeGenerator import *
from buildModel import *

# developing a sample maze in the form of an array with values 0 (blocked cell) or 1 (free cell)
maze=MazeGen(5,5).getMaze()

# feeding sample maze to the program and displaying the maze to be solved
qmaze = Qmaze(maze)
show(qmaze)
plt.show()

# building a neural network model
model = build_model(maze,4)
qtrain(model, maze, epochs=1000, max_memory=8*maze.size, data_size=32)

# playing a game on the built model and displaying the solved maze
play_game(model,qmaze,(0,0))
show(qmaze)
plt.show()