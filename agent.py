import torch
import random
import numpy as np
from game import SnakeGameIA, Direction, Point
from collections import deque

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    
    def _init_(self):
        self.n_games = 0
        self.epsilon = 0 #random
        self.gamma = 0  #discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = None #TODO
        self.trainer = None #TODO
        #TODO model, trainer
        
    def get_state(self, action):
        head = game.snake[0]
        point_l = Point(head.x -20, head.y)
        point_r = Point(head.x +20, head.y)
        point_u = Point(head.x + 20, head.y - 20)
        point_d = Point(head.x + 20, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        state = [
            #Danger Straith
            (dir_r and game.is_colission(point_r)) or
            (dir_r and game.is_colission(point_l))
            (dir_r and game.is_colission(point_u))
            (dir_r and game.is_colission(point_d)),
            
            #Danger rigth
            (dir_u and game.is_colission(point_r)) or
            (dir_d and game.is_colission(point_l))
            (dir_l and game.is_colission(point_u))
            (dir_d and game.is_colission(point_r)),
            
            #Danger left
            (dir_d and game.is_colission(point_r)) or
            (dir_u and game.is_colission(point_l))
            (dir_r and game.is_colission(point_u))
            (dir_l and game.is_colission(point_d)),
        ]
        
        #move directions
        dir_l
        dir_r
        dir_u
        dir_d
        
        #Food
        game.food.x < game.head.x  #food left
        game.food.x > game.head.x  #food rigth
        game.food.y < game.head.y  #food up
        game.food.y > game.head.y #food down
        
        return np.array(state, dtype = int)
    
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long(self):
        if len(self.memory > BATCH_SIZE):
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    def train_short(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        pass
    
def train():
    plot_scores = 0
    plot_mean = 0
    total_score = 0
    record = 0
    agent()
    game = SnakeGameIA()
    
    while True:
        #get old state
        
        state_old = agent.get_state(game)
        
        #get action
        final_move = agent_get_action(state_old)
        
        reward, done, score = game.play_step(final_move)
        
        state_new = agent.get_state(game)
        
        #short train
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            #long train, plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                #agent.model.save()
                
                print('Game', agent.n_games, 'score', score, record, record)
                #TODO plot
        

if __name__ == '__main__':
    train()