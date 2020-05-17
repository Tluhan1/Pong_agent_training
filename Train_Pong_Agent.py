import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from collections import deque 
import numpy as np
import pygame
import random
import cv2

# VARIABLES:

#hyper params
ACTIONS = 3 
GAMMA = 0.99
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
EXPLORE = 500000 
OBSERVE = 50000
REPLAY_MEMORY = 500000
BATCH = 100

# Variables for the game:
speed = 60               
Win_w = 500             
Win_h = 500  
White = (255,255,255)
Black = (0,0,0)
buffer = 40
Player_speed = 0


# CLASS DEFINITION:

class Player:
     
    def __init__(self, xpos, ypos):
        self.xpos = xpos
        self.ypos = ypos
        self.width = 10
        self.height = 70
        self.speed = 0 
        self.ydir = 1
        self.speed = 5
    
    def draw(self):
        P = pygame.Rect(self.xpos, self.ypos, self.width, self.height) 
        pygame.draw.rect(screen,White,P)

    def update(self,Ball_ydir):
        self.speed += 2     
        self.ypos += Ball_ydir*self.speed
        if (self.ypos < 0):
            self.ypos = 0
        if (self.ypos > Win_h-self.height):
            self.ypos = Win_h-self.height
            
    def auto_update(self,Ball_ypos, Ball_height, Ball_xdir):
        if (self.ypos - self.height/2 < Ball_ypos - Ball_height/2 and Ball_xdir == 1):
            self.ypos = self.ypos + self.speed
        if (self.ypos + self.height/2 > Ball_ypos + Ball_height/2 and Ball_xdir == 1):
            self.ypos = self.ypos - self.speed
        if (self.ypos < 0):
            self.ypos = int(Win_h/2)
        if (self.ypos > Win_h - self.height/2):
            self.ypos = int(Win_h/2)
            
    def learn_update(self,action):
        #if move up
        if (action[1] == 1):
            self.ypos = self.ypos - self.speed
        #if move down
        if (action[2] == 1):
            self.ypos = self.ypos + self.speed
        if (self.ypos < 0):
            self.ypos = int(Win_h/2)
        if (self.ypos > Win_h - self.height/2):
            self.ypos = int(Win_h/2) 

class Ball:
     
    def __init__(self, xpos, ypos):
         
        self.xpos = xpos
        self.ypos = ypos
        self.width = 10              
        self.height = 10
        self.xdir = 1
        self.ydir = 1
        self.xspeed = 2        
        self.yspeed = 2 
        
        self.score = 0
        self.score_value = 0
        
        self.right_limit = xpos + self.width/2
        self.left_limit = xpos - self.width/2
        self.up_limit = ypos - self.height/2
        self.down_limit = ypos + self.height/2
    
    def draw(self):
        B = pygame.Rect(self.xpos, self.ypos, self.width, self.height) # instead of ball
        pygame.draw.rect(screen,White,B)
        
    
    def update(self,buffer,player_w,player_ypos): 
        
        self.xpos += self.xdir * self.xspeed
        self.ypos += self.ydir * self.yspeed
        
        # Collision check: with the left and right limit
        # Right side (player side):
        if (int(self.xpos) >= Win_w-buffer-player_w):
            if (int(self.ypos) not in range(player_ypos-35,player_ypos+35)): # Player misses the ball
                self.xpos = Win_w-buffer-player_w
                self.xdir = -1       
                self.score = -1
            elif (int(self.ypos) in range(player_ypos-35,player_ypos+35)):   # Player hits the ball
                self.xpos = Win_w-player_w-buffer
                self.xdir = -1 
                self.score = 1
                self.score_value += 1
        # left side:    
        elif (int(self.xpos) <= player_w):
            self.xpos = player_w
            self.xdir = 1
            return 
                                                              
        # Collision check with top and bottom:
        if (self.ypos <= 0):               # if the ball hits the top:
            self.ypos = 0
            self.ydir = 1
        elif(self.ypos >= Win_h-self.height):   # if the ball hits the bottom:
            self.pos = Win_h-self.height-40
            self.ydir = -1
        return 

    def display_score(self):
        text_font = pygame.font.Font("freesansbold.ttf",14)
#     player1_text = text_font.render(f"{score}",False,White)
        player1_text = text_font.render(f"Score: {self.score_value}",False,White)
        screen.blit(player1_text,(410,480))
        pygame.display.flip()


# FUNCTIONS:
        
def getPresentFrame(Player,Ball):
    pygame.event.pump()
    screen.fill(Black)
    Player.draw()
    Ball.draw()
    image_data = pygame.surfarray.array3d(pygame.display.get_surface())
    screen.blit(pygame.transform.rotate(screen, -90), (0, 0))
    pygame.display.flip()
    Ball.display_score()
    return image_data


def getNextFrame(Player,Ball,action):
    pygame.event.pump()
    score = 0
    screen.fill(Black)    
#     Player.update(Ball.xdir) 
#     Player.auto_update(Ball.ypos, Ball.height, Ball.xdir)
    Player.learn_update(action)         
    Player.draw()
    Ball.update(buffer,int(Player.width),int(Player.ypos))
    Ball.draw()
    image_data = pygame.surfarray.array3d(pygame.display.get_surface())
    screen.blit(pygame.transform.rotate(screen, -90), (0, 0))
    pygame.display.flip()
    Ball.display_score()
    return [score,image_data]

def createGraph():
    #first convolutional layer. bias vector
    #creates an empty tensor with all elements set to zero with a shape
    W_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]))
    b_conv1 = tf.Variable(tf.zeros([32]))

    W_conv2 = tf.Variable(tf.zeros([4, 4, 32, 64]))
    b_conv2 = tf.Variable(tf.zeros([64]))

    W_conv3 = tf.Variable(tf.zeros([3, 3, 64, 64]))
    b_conv3 = tf.Variable(tf.zeros([64]))

    W_fc4 = tf.Variable(tf.zeros([3136, 784]))
    b_fc4 = tf.Variable(tf.zeros([784]))

    W_fc5 = tf.Variable(tf.zeros([784, ACTIONS]))
    b_fc5 = tf.Variable(tf.zeros([ACTIONS]))

    #input for pixel data
    s = tf.placeholder("float", [None, 84, 84, 4])

    #Computes rectified linear unit activation fucntion on  a 2-D convolution given 4-D input and filter tensors. and 
    conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides = [1, 4, 4, 1], padding = "VALID") + b_conv1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides = [1, 2, 2, 1], padding = "VALID") + b_conv2)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides = [1, 1, 1, 1], padding = "VALID") + b_conv3)
    conv3_flat = tf.reshape(conv3, [-1, 3136])
    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)
    fc5 = tf.matmul(fc4, W_fc5) + b_fc5
    return s, fc5


#deep q network. feed in pixel data to graph session 
def trainGraph(inp, out, sess):
    
    # intantiate player and ball objects:
    player = Player(Win_w-buffer-10, Win_h/2-35)
    ball = Ball(Win_w/2-5,Win_h/2-5)
    
    #to calculate the argmax, we multiply the predicted output with a vector with one value 1 and rest as 0
    argmax = tf.placeholder("float", [None, ACTIONS]) 
    gt = tf.placeholder("float", [None]) #ground truth

    #action
    action = tf.reduce_sum(tf.multiply(out, argmax), reduction_indices = 1)
    #cost function we will reduce through backpropagation
    cost = tf.reduce_mean(tf.square(action - gt))
    #optimization fucntion to reduce our minimize our cost function 
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    #create a queue for experience replay to store policies
    D = deque()
    
    #intial frame
    frame = getPresentFrame(player,ball)
       
    #convert rgb to gray scale for processing
    frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
    #binary colors, black or white
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    #stack frames, that is our input tensor
    inp_t = np.stack((frame, frame, frame, frame), axis = 2)

    #saver
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    t = 0
    epsilon = INITIAL_EPSILON
    #training time
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                saver.save(sess, './' + 'pong' + '-dqn', global_step = t)
                pygame.quit()
                break #sys.exit()
            elif event.type == pygame.KEYDOWN:         
                if event.key == pygame.K_ESCAPE:
                    saver.save(sess, './' + 'pong' + '-dqn', global_step = t)
                    pygame.quit()
                    break # sys.exit()
        
        #output tensor
        out_t = out.eval(feed_dict = {inp : [inp_t]})[0]
        #argmax function
        argmax_t = np.zeros([ACTIONS])

        if(random.random() <= epsilon):
            maxIndex = random.randrange(ACTIONS)
        else:
            maxIndex = np.argmax(out_t)
        argmax_t[maxIndex] = 1
        
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #reward tensor if score is positive
        reward_t, frame = getNextFrame(player,ball,argmax_t)
        
        #get frame pixel data
        frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (84, 84, 1))
        #new input tensor
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis = 2)
        
        #add our input tensor, argmax tensor, reward and updated input tensor tos tack of experiences
        D.append((inp_t, argmax_t, reward_t, inp_t1))

        #if we run out of replay memory, make room
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        
        #training iteration
        if t > OBSERVE:

            #get values from our replay memory
            minibatch = random.sample(D, BATCH)
        
            inp_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            inp_t1_batch = [d[3] for d in minibatch]
        
            gt_batch = []
            out_batch = out.eval(feed_dict = {inp : inp_t1_batch})
            
            #add values to our batch
            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))

            #train on that 
            train_step.run(feed_dict = {
                           gt : gt_batch,
                           argmax : argmax_batch,
                           inp : inp_batch
                           })
        
        #update our input tensor the the next frame
        inp_t = inp_t1
        t = t+1
        
        if (t % 10000 == 0):
            print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", maxIndex, "/ REWARD", reward_t, "/ Q_MAX %e" % np.max(out_t))        
        
        #print our where wer are after saving where we are
#         if t % 10000 == 0:
#             saver.save(sess, './' + 'pong' + '-dqn', global_step = t)       


# MAIN:

#initialize game:
pygame.init()
clock = pygame.time.Clock() 
screen = pygame.display.set_mode((Win_w, Win_h)) # instead of SCREEN
pygame.display.set_caption("Thierry's Pong")

#create session
sess = tf.InteractiveSession()

#input layer and output layer by creating graph
inp, out = createGraph()

#train our graph on input and output with session variables
trainGraph(inp, out, sess)