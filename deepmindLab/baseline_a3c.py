import numpy as np
import tensorflow as tf

import gym, time, random, threading, sys

from keras.models import *
from keras.layers import *
from keras import backend as K

from tqdm import trange, tqdm
import csv

from seekavoid_gymlike_wrapper import SeekAvoidEnv

import deepmind_lab

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))


#-- constants
# RUN_TIME = 30
# RUN_TIME = 86000
NUM_ACTIONS = 8
ENV_SHAPE = (84, 84, 3)
THREADS = 8
OPTIMIZERS = 5
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.8
EPS_STOP  = .15
EPS_STEPS = int(1e6)

MIN_BATCH = 32
LEARNING_RATE = 5e-5

LOSS_V = .5                     # v loss coefficient
LOSS_ENTROPY = .01      # entropy coefficient

#---- Preproccessor Parameters

#---------
class Brain:
        train_queue = [ [], [], [], [], [] ]    # s, a, r, s', s' terminal mask
        lock_queue = threading.Lock()

        def __init__(self):
                config = tf.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = 0.9
                self.session = tf.Session(config=config)
                K.set_session(self.session)
                K.manual_variable_initialization(True)

                self.model = self._build_model()
                self.graph = self._build_graph(self.model)

                self.session.run(tf.global_variables_initializer())
                self.default_graph = tf.get_default_graph()

                self.default_graph.finalize()   # avoid modifications

                self.frame_count = 0

                self.csvfile = open("baseline_a3c_history.csv", 'w')
                self.csvwriter = csv.writer(self.csvfile, delimiter=',', quotechar='"')
                self.csvwriter.writerow(['Policy Loss', 'Value Loss', 'Reward', 'Frame Count'])

        def _build_model(self):

                l_input = Input( batch_shape=(None,) + ENV_SHAPE)
                l_conv = Conv2D(32, 4, strides=(2, 2), activation='relu')(l_input)
                l_conv = Conv2D(32, 4, strides=(2, 2), activation='relu')(l_conv)
                l_conv = Conv2D(64, 4, strides=(2, 2), activation='relu')(l_conv)
                l_conv = Conv2D(64, 4, strides=(2, 2), activation='relu')(l_conv)
                l_hidden = Flatten()(l_conv)
                # l_hidden = l_input
                l_hidden = Dense(512, activation='relu')(l_hidden)
                l_hidden = Dense(512, activation='relu')(l_hidden)

                out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_hidden)
                out_value   = Dense(1, activation='linear')(l_hidden)

                model = Model(inputs=[l_input], outputs=[out_actions, out_value])
                model.summary()
                model._make_predict_function()  # have to initialize before threading

                return model

        def _build_graph(self, model):
                s_t = tf.placeholder(tf.float32, shape=(None,) + ENV_SHAPE)
                a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
                r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward

                self.rewards_mean = tf.reduce_mean(r_t)

                p, v = model(s_t)

                log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
                advantage = r_t - v

                loss_policy = - log_prob * tf.stop_gradient(advantage) # Maximize policy
                self.loss_policy = tf.reduce_mean(loss_policy)

                loss_value  = LOSS_V * tf.square(advantage) # minimize value error
                self.loss_value = tf.reduce_mean(loss_value)

                # maximize entropy (regularization) 
                entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10),
                                                       axis=1, keep_dims=True)

                self.loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

                optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
                minimize = optimizer.minimize(self.loss_total)

                return s_t, a_t, r_t, minimize

        def optimize(self):
                if len(self.train_queue[0]) < MIN_BATCH:
                        time.sleep(0)   # yield
                        return

                with self.lock_queue:
                        # more thread could have passed without lock
                        if len(self.train_queue[0]) < MIN_BATCH:
                                return                                                                  # we can't yield inside lock

                        s, a, r, s_, s_mask = self.train_queue
                        self.train_queue = [ [], [], [], [], [] ]

                s = np.asarray(s)
                a = np.vstack(a)
                r = np.vstack(r)
                s_ = np.asarray(s_)
                s_mask = np.vstack(s_mask)

                if len(s) > 5*MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

                v = self.predict_v(s_)
                r = r + GAMMA_N * v * s_mask    # set v to 0 where s_ is terminal state

                s_t, a_t, r_t, minimize = self.graph
                _, policy_loss, value_loss, rewards = self.session.run([minimize,
                                                               self.loss_policy,
                                                               self.loss_value, self.rewards_mean],
                                                              feed_dict={s_t:
                                                                         s,
                                                                         a_t:
                                                                         a,
                                                                         r_t:
                                                                         r})
                self.frame_count += len(s)
                if self.frame_count % (len(s) * 3) == 0:
                    self.model.save_weights("baseline_a3c.h5", overwrite=True)
                    self.csvwriter.writerow([policy_loss, value_loss, rewards, self.frame_count])
                    self.csvfile.flush()

        def train_push(self, s, a, r, s_):
                with self.lock_queue:
                        self.train_queue[0].append(s)
                        self.train_queue[1].append(a)
                        self.train_queue[2].append(r)

                        if s_ is None:
                                self.train_queue[3].append(NONE_STATE)
                                self.train_queue[4].append(0.)
                        else:
                                self.train_queue[3].append(s_)
                                self.train_queue[4].append(1.)

        def predict(self, s):
                with self.default_graph.as_default():
                        p, v = self.model.predict(s)
                        return p, v

        def predict_p(self, s):
                with self.default_graph.as_default():
                        p, v = self.model.predict(s)
                        return p

        def predict_v(self, s):
                with self.default_graph.as_default():
                        p, v = self.model.predict(s)
                        return v

#---------
frames = 0
class Agent:
        def __init__(self, eps_start, eps_end, eps_steps):
                self.eps_start = eps_start
                self.eps_end   = eps_end
                self.eps_steps = eps_steps

                self.memory = []        # used for n_step return
                self.R = 0.

        def getEpsilon(self):
                if(frames >= self.eps_steps):
                        return self.eps_end
                else:
                        return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps       # linearly interpolate

        def act(self, s):
                eps = self.getEpsilon()
                global frames; frames = frames + 1

                if random.random() < eps:
                        return random.randint(0, NUM_ACTIONS-1)

                else:
                        s = np.array([s])
                        p = brain.predict_p(s)[0]

                        # a = np.argmax(p)
                        a = np.random.choice(NUM_ACTIONS, p=p)

                        return a

        def train(self, s, a, r, s_):
                def get_sample(memory, n):
                        s, a, _, _  = memory[0]
                        _, _, _, s_ = memory[n-1]

                        return s, a, self.R, s_

                a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
                a_cats[a] = 1

                self.memory.append( (s, a_cats, r, s_) )

                self.R = ( self.R + r * GAMMA_N ) / GAMMA

                if s_ is None:
                        while len(self.memory) > 0:
                                n = len(self.memory)
                                s, a, r, s_ = get_sample(self.memory, n)
                                brain.train_push(s, a, r, s_)

                                self.R = ( self.R - self.memory[0][2] ) / GAMMA
                                self.memory.pop(0)

                        self.R = 0

                if len(self.memory) >= N_STEP_RETURN:
                        s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
                        brain.train_push(s, a, r, s_)

                        self.R = self.R - self.memory[0][2]
                        self.memory.pop(0)

        # possible edge case - if an episode ends in <N steps, the computation is incorrect

#---------
class Environment(threading.Thread):
        stop_signal = False

        def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
                threading.Thread.__init__(self)

                self.render = render
                self.env = SeekAvoidEnv()
                # self.env = gym.make("CartPole-v1")
                self.agent = Agent(eps_start, eps_end, eps_steps)

        def runEpisode(self):
                s = self.env.reset()

                R = 0
                while True:
                        time.sleep(THREAD_DELAY) # yield 

                        if self.render: self.env.render()

                        a = self.agent.act(s)
                        s_, r, done, info = self.env.step(a)
                        # print(self.ident, info['step'])

                        if done: # terminal state
                                s_ = None

                        self.agent.train(s, a, r, s_)

                        s = s_
                        R += r

                        if done or self.stop_signal:
                                break

                print(R)

        def run(self):
                while not self.stop_signal:
                        self.runEpisode()

        def stop(self):
                self.stop_signal = True

#---------
class Optimizer(threading.Thread):
        stop_signal = False

        def __init__(self):
                threading.Thread.__init__(self)

        def run(self):
                while not self.stop_signal:
                        brain.optimize()

        def stop(self):
                self.stop_signal = True

#-- main
if __name__ == "__main__":
    env_test = Environment(render=False, eps_start=0., eps_end=0.)
    NONE_STATE = np.zeros(ENV_SHAPE)

    brain = Brain() # brain is global in A3C

    envs = [Environment() for i in range(THREADS)]
    opts = [Optimizer() for i in range(OPTIMIZERS)]

    for o in opts:
            o.start()

    for e in envs:
            e.start()

    num_frames = 1.6 * 1e7
    tq = tqdm(total=num_frames)
    last_frame = 0
    while brain.frame_count < num_frames:
        # sys.stderr.write(str(brain.frame_count) + "\n")
        tq.update(brain.frame_count - last_frame)
        last_frame = brain.frame_count
        time.sleep(60)
    tq.close()

    for e in envs:
            e.stop()
    for e in envs:
            e.join()

    for o in opts:
            o.stop()
    for o in opts:
            o.join()
