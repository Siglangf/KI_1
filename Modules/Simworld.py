# Requirements
# 1. Understands game states and the operators that convert one game state to another.
# 2. Produces initial game states.
# 3. Generates child states from parent states using the legal operators of the domain
# 4. Recognizes final (winning, losing and neutral) states.

# Methods
# produce-initial-state
#generate-all- child-states-of-the-current-state
# is-the-current-state-a-final-state
import networkx as nx
import matplotlib.pyplot as plt
from random import randint
import numpy as np
NEIGHBOAR_MAPPING = {(0,1):'UP',(0,-1):'DOWN',(-1,0):'LEFT',(1,0):'RIGHT', (-1,1):'LEFT_DIAG',(1,-1):'RIGHT_DIAG'}
COLOR_MAP = {True: "blue",False:"white"}
#class Simworld(self,)
'''
def __init__(self, game_type):
    print(f"initialized {game_type}")

def produce_initial_state(self):
    print(f"Not implemented")

def generate_all_childstates_of_current_state(self, state):
    print("Not implemented")
#
def is_final_state(self, state):
    print("Not implemented")
'''

class Peg_Solitaire:
    def __init__(self, board_type, size,open_cells=1):
        self.board_type = board_type
        self.size=size
        self.open_cells = open_cells
        self.board = Board(board_type,size,open_cells)  
    def is_final_state(self):
        #If there is a new win
        return len(self.generate_all_childstates()) == 0
    def collect_reward(self):
      '''How shall reward be given?'''
      remaining_cells = 0
      for cell in np.hstack(self.board.cells):
          if cell.state:
              remaining_cells+=1
      win  = remaining_cells==1
      return win
    def generate_all_childstates(self):
        action_space = []
        empty_cells = self.board.get_empty_cells()
        for cell in empty_cells:
            for pos, neig in cell.neighboars.items():
                if neig.state==False:
                    continue
                if pos in neig.neighboars.keys():
                    if neig.neighboars[pos].state==True:
                        action_cells = (cell,neig,neig.neighboars[pos])
                        action_space.append(action_cells)
        return action_space                  
    def do_action(self,action_cells):
        action_cells[0].state=True
        action_cells[1].state=False
        action_cells[2].state=False
    def action_to_string(self,action_cells):
        return f"Flytte brikke på posisjon {action_cells[2]} til posisjon {action_cells[0]} of fjerne {action_cells[1]}"
             
class Cell:
    def __init__(self,state:bool,position):
        self.state = state
        self.position = position
    def invert_cell(self):
        self.state = not self.state
    def _repr__(self):
        state = "Pegged" if self.state else "Empty"
        return f"{state} cell at position ({self.position[0]},{self.position[1]})"
    def __str__(self):
        return str(self.position)
        
class Board:
    def __init__(self,board_type,size,open_cells):
        self.board_type = board_type
        self.size = size
        self.open_cells = open_cells
        center_X = size//2
        center_Y = size//2 if board_type=="diamond" else size//4
        #Initializing cell object
        if board_type=="triangle":
            self.cells = np.array([[Cell(True,[j,i]) for i in range(size-j)] for j in range(self.size)],dtype=object)
        elif board_type=="diamond":
            self.cells = np.array([[Cell(True,[i,j]) for i in range(self.size)] for j in range(self.size)])
        else:
            raise ValueError("Board type must be 'triangle' or 'diamond'")
        
        self.connect_adjacent()
        #Make sure at least one is in the center
        self.cells[center_X][center_Y].state = False
        #Choose the remaining empty spots
        for cell in np.random.choice(np.hstack(np.delete(self.cells,[center_X,center_Y])),self.open_cells-1, replace=False):
            cell.state=False
    def get_empty_cells(self):
        return [cell for cell in np.hstack(self.cells) if cell.state==False]
    def connect_adjacent(self):
        '''Connect adjecent cells to each other in cell.neighboars'''
        cells = self.cells
        for i in range(len(cells)):
            for j in range(len(cells[i])):
                cells[i][j].neighboars = {}
                for delta,position in NEIGHBOAR_MAPPING.items():
                    adjx = i+delta[0]
                    adjy = j+delta[1]
                    #Make sure avoiding negative index referances
                    if adjx<0 or adjy<0:
                        continue
                    try:
                        cells[i][j].neighboars[position] = cells[adjx][adjy]
                    except:
                        continue
    def to_array(self):
      l =[]
      for row in self.cells:
        row = [int(cell.state) for cell in row]
        l.append(row)
      return np.array(l) 
    def update_from_array(self, array):
    def visualize(self):
        cells = np.hstack(self.cells)
        G = nx.Graph()
        G.add_nodes_from(cells)
        for cell in cells:
            for pos, neighboar in cell.neighboars.items():
                G.add_edge(cell, neighboar)
        
        positions = {cell:[cell.position[0],cell.position[1]] for cell in cells}
        colors= []
        for node in G:
            colors.append(COLOR_MAP[node.state])
        fig = plt.figure()
        nx.draw(G,pos=positions,ax=fig.add_subplot(),node_color=colors)
        return fig

def visualize_state(Board):
  cells = np.hstack(Board.cells)
  G = nx.Graph()
  G.add_nodes_from(cells)
  for cell in cells:
      for pos, neighboar in cell.neighboars.items():
          G.add_edge(cell, neighboar)
  
  positions = {cell:[cell.position[0],cell.position[1]] for cell in cells}
  colors= []
  plt.ioff()
  for node in G:
      colors.append(COLOR_MAP[node.state])
  fig = plt.figure()
  nx.draw(G,pos=positions,ax=fig.add_subplot(),node_color=colors)
  return fig