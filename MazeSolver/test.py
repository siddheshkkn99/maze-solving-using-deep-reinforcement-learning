# ---------------program to solve a maze  using deep reinforced learning technique---------------------------------

# ---------------------------------------------------------------------------------
# importing maze generator and solver programs
from mazeSolve import *
from mazeGenerator import *
# ---------------------------------------------------------------------------------
# developing a sample maze in the form of an array with values 0(blocked cell) or 1(free cell)
print("Enter no. of rows and columns in maze")
nrows,ncols = map(int,input().split())
maze=MazeGen(nrows,ncols).getMaze()
# ----------------------------------------------------------------------------------
'''
# sample maze
maze = np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  1.],
    [ 1.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  1.],
    [ 1.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    [ 1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.]
])
'''
# feeding sample maze to the program
qmaze = Qmaze(maze)

# displaying input maze
show(qmaze)
plt.show()

# building a neural network model
model = build_model(maze)
qtrain(model, maze, epochs=1000, max_memory=8*maze.size, data_size=32)

# playing a game on the built model
play_game(model,qmaze,(0,0))
# displaying solved maze
show(qmaze)
plt.show()
