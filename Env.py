# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


max_requests = 15 # given in the problem statement,  driver can have max 15 requests
episode_time_in_hrs = 24*30   # 24 hrs x 30 days



class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [[i,j] for i in range(m) for j in range(m) if i!=j or [i,j]==[0,0]]
        self.state_space = [[i,j,k] for i in range(m) for j in range(t) for k in range(d)]
        self.state_init = random.choice(self.state_space)

        self.reset()



    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""

        state_encod = np.zeros((m+t+d))
        state_encod[state[0]] = 1
        state_encod[m+ int(state[1])]=1
        state_encod[m+t+ int(state[2])]=1


        return state_encod


    # Use this function if you are using architecture-2 (state and action  as input vector)
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        state_encod = np.zeros(m+t+d+m+m)
        state_encod.reshape(1,m+t+d+m+m)
        state_encod[state[0]] = 1
        state_encod[m+np.int(state[1])] = 1
        state_encod[m+np.int(t+state[2])] = 1
        state_encod[m+t+d+action[0]] = 1
        state_encod[m+t+d+m+action[1]] = 1
        return state_encod



    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)      # location A
        elif location == 1: 
            requests= np.random.poisson(12)      # location B
        elif location == 2:
            requests= np.random.poisson(4)       # location C
        elif location == 3:
            requests= np.random.poisson(7)       # location D
        else:
            requests = np.random.poisson(8)      # location E


        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        
        actions.append([0,0])          # Appending/Adding  (0,0) to actions when driver is offline
        return actions   



 
    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        start_loc, time, day = state            # state is in the form of triplet i.e, (loc,time,day)
        pick_up, drop = action                  # action is in the form of tuple(p,q)
        
        if pick_up == 0 and drop ==0:      # driver is offline, not accepting requests scenario
            return -5                      # Reward of - C will be given because of Per hour fuel and other costs
        else:
           # rewards when driver is online and pickup and drop happens
           time_elapsed_till_pickup = Time_matrix[start_loc,pick_up,int(time), int(day)]      # Time Matrix is a 4-D matrix, 𝑇𝑖𝑚𝑒 − 𝑚𝑎𝑡𝑟𝑖𝑥[𝑠𝑡𝑎𝑟𝑡 − 𝑙𝑜𝑐][𝑒𝑛𝑑 − 𝑙𝑜𝑐][h𝑜𝑢𝑟 o𝑓 𝑡h𝑒 d𝑎𝑦] [𝑑𝑎𝑦 𝑜𝑓 𝑡h𝑒 𝑤𝑒𝑒𝑘]
           time_next = int((time + time_elapsed_till_pickup) % t )
           day_next = int((day + (time + time_elapsed_till_pickup)//t) % d)
           reward = (R*Time_matrix[pick_up,drop, time_next,day_next]) - C*(Time_matrix[pick_up,drop,time_next,day_next] + Time_matrix[start_loc, pick_up, time, day])
           return reward
           



    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        start_loc, time, day = state
        pick_up, drop = action
        done = False   # teminal state or episode ends = False initially 
        
        if pick_up==0 and drop==0:       # driver is offline, not accepting request, no ride action (0,0)
            time_elapsed = 1             # The no- ride action just moves the time component by 1 hour
            self.total_time += time_elapsed
        
        else:
           time_elapsed_till_pickup = Time_matrix[start_loc, pick_up, int(time), int(day)] # Time Matrix is a 4-D matrix, 𝑇𝑖𝑚𝑒 − 𝑚𝑎𝑡𝑟𝑖𝑥[𝑠𝑡𝑎𝑟𝑡 − 𝑙𝑜𝑐][𝑒𝑛𝑑 − 𝑙𝑜𝑐][h𝑜𝑢𝑟 o𝑓 𝑡h𝑒 d𝑎𝑦] [𝑑𝑎𝑦 𝑜𝑓 𝑡h𝑒 𝑤𝑒𝑒𝑘]
           pickup_time = int((time + time_elapsed_till_pickup) % t )   # actual time when driver reaches pick up point
           pickup_day = int((day + (time + time_elapsed_till_pickup)//t) % d)
           time= pickup_time
           day= pickup_day

           
           time_elapsed = Time_matrix[pick_up,drop,time,day]
           self.total_time += (time_elapsed + time_elapsed_till_pickup)
           
        time_next = int((time + time_elapsed) % t)
        day_next = int((day + (time + time_elapsed)//t)%d)
        
        # check terminal state is reached or not
        if (self.total_time >= episode_time_in_hrs):
            done = True          # reached terminal state
        else:
            done = False
        next_state = [drop, time_next, day_next]
        
        return next_state, done




    def reset(self):
        self.total_time=0 # initilaise it to 0, beginning of episode
        self.total_rewards=0 
        self.action_size = m*(m-1) + 1
        return self.state_init
    

