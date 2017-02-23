import numpy as np

# R matrix
R = np.matrix([ [-1,-1,-1,-1,0,-1],
		[-1,-1,-1,0,-1,100],
		[-1,-1,-1,0,-1,-1],
		[-1,0,0,-1,0,-1],
		[-1,0,0,-1,-1,100],
		[-1,0,-1,-1,0,100] ])

# Q matrix
Q = np.matrix(np.zeros([6,6]))

# Gamma (learning parameter).
gamma = 0.8

# Initial state. (Usually to be chosen at random)
initial_state = 1

# This function returns all available actions in the state given as an argument
def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

# Get available actions in the current state
available_act = available_actions(initial_state) 

# This function chooses at random which action to be performed within the range 
# of all the available actions.
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act,1))
    return next_action

# Sample next action to be performed
action = sample_next_action(available_act)

# This function updates the Q matrix according to the path selected and the Q 
# learning algorithm
def update(current_state, avaliable_action, gamma):
    
    
    
    max_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)
     
    while not (max_index in avaliable_action):
        Q[current_state, max_index] = 0
        max_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
        if max_index.shape[0] > 1:
            max_index = int(np.random.choice(max_index, size = 1))
        else:
            max_index = int(max_index)
    #print current_state, max_index  
    
   
    
    next_state = max_index
    
    avaliable_action = available_actions(next_state)
    max_index2 = np.where(Q[next_state,] == np.max(Q[next_state,]))[1]
    if max_index2.shape[0] > 1:
        max_index2 = int(np.random.choice(max_index2, size = 1))
    else:
        max_index2 = int(max_index2)
     
    while not (max_index2 in avaliable_action):
        Q[next_state, max_index2] = 0
        max_index2 = np.where(Q[next_state,] == np.max(Q[next_state,]))[1]
        if max_index2.shape[0] > 1:
            max_index2 = int(np.random.choice(max_index2, size = 1))
        else:
            max_index2 = int(max_index2)
    # Q learning formula
   # print next_state, max_index2
    Q[current_state, max_index] = R[current_state, max_index] + gamma * Q[next_state, max_index2]


#-------------------------------------------------------------------------------
# Training

# Train over 10 000 iterations. (Re-iterate the process above).
for i in range(10000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
   
    update(current_state, available_act, gamma)
    
# Normalize the "trained" Q matrix
print("Trained Q matrix:")
print(Q/np.max(Q)*100)

#-------------------------------------------------------------------------------
# Testing

# Goal state = 5
# Best sequence path starting from 2 -> 2, 3, 1, 5

current_state = 2
steps = [current_state]

while current_state != 5:

    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
    
    steps.append(next_step_index)
    current_state = next_step_index

# Print selected sequence of steps
print("Selected path:")
print(steps)