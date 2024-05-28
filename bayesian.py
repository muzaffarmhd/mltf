JustPaste.it
User avatar
Cutting Player @cuttingplayer · 1s
Public 
```Python
import numpy as np
grid_size = (3,4)
terminals = {(0,3):1,(1,3):-1}
walls = [(1,1)]
gamma = 0.9
theta = 1e-4


U = np.zeros(grid_size)
rewards = np.full(grid_size,-0.04)
for terminal, reward in terminals.items():
 rewards[terminal]=reward
actions = ["up","down","left","right"]
action_effects = {
 "up":(-1,0),
 "down":(1,0),
 "left":(0,-1),
 "right":(0,1)
}
probabilities = {
 "up":[0.8,0.1,0.1],
 "down":[0.8,0.1,0.1],
 "left":[0.8,0.1,0.1],
 "right":[0.8,0.1,0.1]
}
direction_prob = {
 "up":["up","left","right"],
 "down":["down","right","left"],
 "left":["left","down","up"],
 "right":["right","up","down"]
}
def is_valid_state(state):
 r,c = state
 return 0 <= r < grid_size[0] and 0 <= c < grid_size[1] and (r,c) not in walls
def get_next_state(state,action):
 effects = action_effects[action]
 next_state = (state[0]+effect[0],state[1]+effect[1])
 return next_state if is_valid_state(next_state) else state
def get_expected_utility(state,action,U)
 return sum(probabilities[action][i]*U[get_next_state(state,direction)] for i, direction in enumerate(direction_prob[action]))
def value_iteration(U,rewards, gamma, theta):
 while True:
 delta = 0
 U_new = np.copy(U)
 for r in range(grid_size[0]):
  for c in range(grid_size[1]):
  state = (r,c)
  if state in terminals or state in walls:
  continue
 new_utility = rewards[state]+gamma*max(get_expected_utility(state,action) for action in actions)
 delta = max(delta, abs(new_utility-U_new[state]))
 U_new[state]=new_utility
 if delta < theta:
 break
 U=U_new
 return U
def extract_policy(U,rewards,gamma):
 policy={}
 for r in range(grid_size[0]):
 for c in range(grid_size[1]):
 state=(r,c)
 if state in terminals or state in walls
 continue
 policy[state]=max(actions, key = lambda action: rewards[state]+gamma*get_expected_utility(state,action,U))
 return policy
U_value = value_iteration(U,rewards, gamma, theta)
policy = extract_policy(U_value, rewards, gamma)
print(policy)
```


## 8-Puzzle
```Python
import heapq


# Node class to represent states in the puzzle
class Node:
    def __init__(self, state, parent=None, action=None, cost=0, heuristic=0):
        self.state = state  # Current puzzle configuration
        self.parent = parent  # Node from which this state was reached
        self.action = action  # Move that led to this state
        self.cost = cost  # Cost to reach this state from the start
        self.heuristic = heuristic  # Estimated cost to reach the goal


    # Method to compare nodes for priority queue (based on cost + heuristic)
    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


# Function to generate valid successor states from a given state
def successors(state):
    successors = []  # List to store valid moves
    empty = [(i, j) for i in range(3) for j in range(3) if state[i][j] == 0][0]  # Find the empty tile
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Define possible moves


    for move in moves:
        newpos = (empty[0] + move[0], empty[1] + move[1])
        if 0 <= newpos[0] < 3 and 0 <= newpos[1] < 3:  # Check if the move is within bounds 
            new_state = list(map(list, state))  # Copy the state to modify
            new_state[empty[0]][empty[1]], new_state[newpos[0]][newpos[1]] = new_state[newpos[0]][newpos[1]], new_state[empty[0]][empty[1]]  # Swap tiles
            successors.append((move, tuple(map(tuple, new_state)), 1))  # Store the move, new state, and cost 


    return successors


# Heuristic function: Number of misplaced tiles 
def heuristic(state, goal_state, cost_so_far):
    misplaced_tiles = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != goal_state[i][j] and state[i][j] != 0:
                misplaced_tiles += 1
    return misplaced_tiles


# A* search algorithm
def astar(start_state, goal_state, successors, heuristic):
    f = []  # Priority queue for open nodes
    heapq.heappush(f, Node(start_state, None, None, 0, heuristic(start_state, goal_state, 0)))
    explored = set()  # Set to keep track of visited states


    while f:
        current_node = heapq.heappop(f)
        current_state = current_node.state


        if current_state == goal_state:
            return goal_path(current_node)  # Goal found!


        explored.add(current_state)


        for action, next_state, stepcost in successors(current_state):  # Changed from list(current_state) for efficiency
            if next_state not in explored:
                next_node = Node(next_state, current_node, action, current_node.cost + stepcost, heuristic(next_state, goal_state, current_node.cost + stepcost))
                heapq.heappush(f, next_node)


    return None  # Goal not found


# Reconstruct the path from the goal node back to the start
def goal_path(node):
    path = []
    while node:
        path.append((node.action, node.state))
        node = node.parent
    path.reverse()  # Reverse to get the path from start to goal
    return path


# Example usage
start_state = (
    (1, 2, 3),
    (4, 0, 5),
    (6, 7, 8)
)
goal_state = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 0)
)
path = astar(start_state, goal_state, successors, heuristic)
if path:
    print("Goal reached:", path) 
else:
    print("Goal not reached!") 

 


```
## Water Jug
```Python
import heapq


class State:
    def __init__(self, jug1, jug2):
        self.jug1 = jug1
        self.jug2 = jug2
    def __eq__(self, other):
        return self.jug1 == other.jug1 and self.jug2 == other.jug2


    def __hash__(self):
        return hash((self.jug1, self.jug2))


    def __repr__(self):
        return f"State({self.jug1}, {self.jug2})"


class Node:
    def __init__(self, state, parent=None, action=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.heuristic = heuristic


    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


def heuristic(state):
    goal = 2
    return abs(state.jug1 - goal) + abs(state.jug2 - goal)


def goal_test(state):
    goal = 2
    return state.jug1 == goal or state.jug2 == goal


def successors(state):
    successors = []
    actions = [
        (jug1_capacity, state.jug2),  
        (state.jug1, jug2_capacity),  
        (0, state.jug2),              
        (state.jug1, 0),              
        (max(0, state.jug1 - (jug2_capacity - state.jug2)), min(jug2_capacity, state.jug1 + state.jug2)),
        (min(jug1_capacity, state.jug1 + state.jug2), max(0, state.jug2 - (jug1_capacity - state.jug1)))
    ]
    for action in actions:
        next_state = State(*action)
        successors.append((action, next_state, 1)) 
    return successors


def astar_search(start_state, goal_test, successors, heuristic):
    frontier = []  
    explored = set()  


    heapq.heappush(frontier, Node(start_state, None, None, 0, heuristic(start_state)))


    while frontier:
        current_node = heapq.heappop(frontier)
        current_state = current_node.state


        if goal_test(current_state):
            return get_path(current_node)


        explored.add(current_state)


        for action, next_state, step_cost in successors(current_state):
            if next_state not in explored:
                total_cost = current_node.cost + step_cost
                next_node = Node(next_state, current_node, action, total_cost, heuristic(next_state))
                heapq.heappush(frontier, next_node)


    return None  


def get_path(node):
    path = []
    current_node = node
    while current_node:
        path.append((current_node.action, current_node.state))
        current_node = current_node.parent
    return list(reversed(path)) 


initial_state = State(0, 0)  
goal = 2  
jug1_capacity = 4
jug2_capacity = 3

 


path = astar_search(initial_state, goal_test, successors, heuristic)


if path: 
    print("Solution path:")
    for action, state in path:
        print(f"Action: {action}, Jug1: {state.jug1}, Jug2: {state.jug2}")
else:
    print("No solution found.") 


```


## Tic-Tac-Toe
```Python
def print_board(board):
    for i in range(0, 9, 3):
        print(" | ".join(board[i:i + 3]))
        if i < 6: print('-' * 6)


def is_winner(board, player):
    wins = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)]
    return any(all(board[i] == player for i in pos) for pos in wins)


def minimax(board, player, is_max):
    if is_winner(board, 'X'): return -1
    if is_winner(board, 'O'): return 1
    if ' ' not in board: return 0
    best = float('-inf') if is_max else float('inf')
    for move in [i for i, v in enumerate(board) if v == " "]:
        board[move] = player if is_max else ('O' if player == 'X' else 'X')
        score = minimax(board, player, not is_max)
        board[move] = ' '
        best = max(best, score) if is_max else min(best, score)
    return best


def get_best_move(board, player):
    best_move, best_score = None, float('-inf')
    for move in [i for i, v in enumerate(board) if v == " "]:
        board[move] = player
        score = minimax(board, player, False)
        board[move] = ' '
        if score > best_score:
            best_score, best_move = score, move
    return best_move


def play_game():
    board, current_player = [" "] * 9, 'X'
    while not (is_winner(board, 'X') or is_winner(board, 'O') or ' ' not in board):
        print_board(board)
        move = int(input('Enter move (0-8): ')) if current_player == 'X' else get_best_move(board, 'O')
        board[move], current_player = current_player, 'O' if current_player == 'X' else 'X'
    print_board(board)
    print('Player X wins!' if is_winner(board, 'X') else 'Player O wins!' if is_winner(board, 'O') else 'It\'s a draw!')


play_game()


```


## Bayes
```Python
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import pandas as pd


# Sample data
data = pd.DataFrame(data={'Rain': ['No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No'],
                          'TrafficJam': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No'],
                          'ArriveLate': ['Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No']})


# Define the Bayesian network structure
model = BayesianNetwork([('Rain', 'TrafficJam'), ('TrafficJam', 'ArriveLate')])


# Fit the model to the data using Bayesian Parameter Estimation
model.fit(data, estimator=BayesianEstimator)


# Print conditional probability distributions
cpds = model.get_cpds()
for cpd in cpds:
    print("CPD for variable:", cpd.variable)
    print(cpd)


# Perform inference
inference = VariableElimination(model)
query_result = inference.query(variables=['ArriveLate'], evidence={'Rain': 'Yes'})
print(query_result)
```



Graph A*

import heapq

class GraphNode:
    def __init__(self, vertex, parent=None, cost=0, heuristic=0):
        self.vertex = vertex
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def astar_graph(graph, start, goal, heuristics):
    f = []
    heapq.heappush(f, GraphNode(start, None, 0, heuristics[start]))
    explored = set()
    
    while f:
        current_node = heapq.heappop(f)
        current_vertex = current_node.vertex
        
        if current_vertex == goal:
            return reconstruct_path(current_node)
        
        explored.add(current_vertex)
        
        for neighbor, weight in graph[current_vertex]:
            if neighbor not in explored:
                next_node = GraphNode(neighbor, current_node, current_node.cost + weight, heuristics[neighbor])
                heapq.heappush(f, next_node)
    
    return None

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.vertex)
        node = node.parent
    path.reverse()
    return path

# Heuristic values from the given graph
heuristics = {
    'S': 9,
    'A': 8,
    'B': 6,
    'C': 7,
    'D': 6,
    'E': 6,
    'F': 3,
    'G': 0
}

# Graph structure from the given graph
graph = {
    'S': [('A', 1), ('B', 5)],
    'A': [('S', 1), ('B', 2), ('C', 4), ('D', 4)],
    'B': [('S', 5), ('A', 2), ('D', 6), ('E', 3)],
    'C': [('A', 4), ('F', 5)],
    'D': [('A', 4), ('B', 6), ('F', 6)],
    'E': [('B', 3), ('G', 10)],
    'F': [('C', 5), ('D', 6), ('G', 5)],
    'G': []
}

start = 'S'
goal = 'G'
path = astar_graph(graph, start, goal, heuristics)

if path:
    print("Path found:", path)
else:
    print("Path not found!")

1 visits · 1 online
  
© 2024 JustPaste.it
Account
Terms
Privacy
Cookies
Blog
About