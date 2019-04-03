class MazeGen:
    # constructor function
    def __init__(self,ncols,nrows):
        # 0 - blocked
        # 1 - free
        # initialising maze with all blocked cells 
        self.maze=np.array([ [0 for x in range(nrows)] for y in range(ncols) ])

        # setting rat(agent) and cheese(target) cells
        self.rat=(0,0)
        self.cheese=(ncols-1,nrows-1)
        
        # freeing rat and cheese cells
        self.maze[self.rat]=1
        self.maze[self.cheese]=1

    # function to make a move
    def act(self,move):
        row,col=self.rat
        if move=='l':col-=1
        elif move=='r':col+=1
        elif move=='u':row-=1
        elif move=='d':row+=1
        else: return -1
        self.rat=(row,col)

    # function to validate a move by checking if the rat movement 
    # is restriceted by the sides of the maze or not
    def validAction(self):
        # all possible moves: left, right, up and down
        validActions=['l','r','u','d']
        nrow,ncol=self.maze.shape
        row,col=self.rat
        if row==0: validActions.remove('u')
        elif row==nrow-1: validActions.remove('d')
        if col==0: validActions.remove('l')
        elif col==ncol-1: validActions.remove('r')
        return validActions

    # function to traverse the maze randomly until target is reached
    def traverse(self):
        while self.rat!=self.cheese :
             # finding all valid moves possible
            actionList=self.validAction()
            if len(actionList)==0: return -1
            
            # making a random move
            action=random.choice(actionList)
            self.act(action)
            
            # setting visited cells as freecells 
            currRow,currCol=self.rat
            self.maze[(currRow,currCol)]=1
            
    def getMaze(self):
        print("Generating maze ......")
        self.traverse()
        nrow,ncol=self.maze.shape  
        return self.maze

print("Enter no of rows and columns of maze")
nrow,ncol=map(int,input().split())
newMaze=MazeGen(nrow,ncol)
print(newMaze.getMaze())
