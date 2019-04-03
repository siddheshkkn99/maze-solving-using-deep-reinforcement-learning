import numpy as np
import random

class MazeGen:
    def __init__(self,ncols,nrows):
        # 0 - blocked
        # 1 - free
        # initialising maze with all blocked cells 
        self.maze=np.array([[0]*ncols]*nrows)
        # setting rat(agent) and cheese(target) cells
        self.rat=(0,0)
        self.cheese=(ncols-1,nrows-1)
        # freeing rat and cheese cells
        self.maze[self.rat]=1.0
        self.maze[self.cheese]=1.0
        # storing all 
        self.freeCells=[]
        print(self.freeCells)

    def traverse(self):
        while self.rat!=self.cheese:
            currRow,currCol=self.rat
            action=validAction()
            if len(action)==0: return -1
            self.freeCells.append(rat)
    
    def validAction(self):
        
        return actions
        
    def getMaze(self):
        return self.maze

newMaze=MazeGen()
print(newMaze.initmaze(5,5))
