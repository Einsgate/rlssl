import numpy as np
import scipy as sp
import pickle
import copy
import random
import time
import math
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
import matlab.engine


UNIT = 40   # pixels
MAZE_H = 9  # grid height
MAZE_W = 9  # grid width

## Microphone configuration
maxDis = 0.2828
N = 4
fs = 16000
width = int(math.ceil(maxDis / 340 * fs) * 2 + 1)
npairs = int(sp.misc.comb(N, 2))

class Maze(tk.Tk, object):
    def __init__(self, ori, des):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r', 'ul', 'ur', 'dl', 'dr']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()
        self.des = copy.copy(des)
        self.ori = copy.copy(ori)
        self.gcc_width = width
        self.npairs = npairs
        self.n_features = int(self.gcc_width * self.npairs) + 2

        self.load_env()


    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # # hell
        # hell1_center = origin + np.array([UNIT * 2, UNIT])
        # self.hell1 = self.canvas.create_rectangle(
        #     hell1_center[0] - 15, hell1_center[1] - 15,
        #     hell1_center[0] + 15, hell1_center[1] + 15,
        #     fill='black')
        # # hell
        # hell2_center = origin + np.array([UNIT, UNIT * 2])
        # self.hell2 = self.canvas.create_rectangle(
        #     hell2_center[0] - 15, hell2_center[1] - 15,
        #     hell2_center[0] + 15, hell2_center[1] + 15,
        #     fill='black')

        # create oval
        oval_center = origin + UNIT * 8
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        #self.update()
        #time.sleep(0.05)
        self.canvas.delete(self.rect)
        self.canvas.delete(self.oval)

        #sink = np.array([self.des[0] * UNIT + 20, self.des[1] * UNIT + 20])
        #origin = np.array([self.ori[0] * UNIT + 20, self.ori[1] * UNIT + 20])

        # Randomly choose origin
        while 1:
            ori = random.choice(self.ori)
            des = random.choice(self.des)
            origin = np.array([ori[0] * UNIT + 20, ori[1] * UNIT + 20])
            sink = np.array([des[0] * UNIT + 20, des[1] * UNIT + 20])
            if not (sink[0] == origin[0] and sink[1] == origin[1]):
                break

        # Randomize origin
        # while 1:
        #     rand_x = np.random.randint(MAZE_W, size=1)
        #     rand_y = np.random.randint(MAZE_H, size=1)
        #     origin = np.array([rand_x[0] * UNIT + 20, rand_y[0] * UNIT + 20])
        #     #sink = np.array([rand_x[1] * UNIT + 20, rand_y[1] * UNIT + 20])
        #     if not (rand_x[0] == self.des[0] and rand_y[0] == self.des[1]):   # In case origin and sink are the same position
        #         break

        # Re-create origin and sink
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        self.oval = self.canvas.create_oval(
            sink[0] - 15, sink[1] - 15,
            sink[0] + 15, sink[1] + 15,
            fill='yellow')

        # Get gcc
        xdi, ydi = self.d_index()
        xsi, ysi = self.s_index()
        gcc = self.GCC[xdi][ydi][xsi][ysi]

        # Get position
        xs, ys = self.s_position()

        # return observation
        return np.append(gcc, [xs, ys])

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 4:   # left up
            if s[0] > UNIT and s[1] > UNIT:
                base_action[0] -= UNIT
                base_action[1] -= UNIT
        elif action == 5:   # right up
            if s[0] < (MAZE_W - 1) * UNIT and s[1] > UNIT:
                base_action[0] += UNIT
                base_action[1] -= UNIT
        elif action == 6:   # left down
            if s[0] > UNIT and s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
                base_action[0] -= UNIT
        elif action == 7:   # right down
            if s[0] < (MAZE_W - 1) * UNIT and s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
                base_action[0] += UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        # p_ = self.canvas.coords(self.rect)  # next position

        # Get gcc (next state)
        xsi, ysi = self.s_index()
        xdi, ydi = self.d_index()
        xs, ys = self.s_position()
        s_ = np.append(self.GCC[xdi][ydi][xsi][ysi], [xs, ys])

        # print(self.GCC[4][4])
        # print(self.GCC[5][3])
        # time.sleep(100)
        #gcc = list(s_)
        #print("current position: %d %d, argmax gcc: %d" %(xsi, ysi, gcc.index(max(gcc))))

        #s_ = np.array(gcc)

        # Reward function
        #reward = -(math.sqrt(pow(xsi - xdi, 2) + pow(ysi - ydi, 2)))
        # xx, yy= abs(xsi - xdi), abs(ysi - ydi)
        # layer = abs(xx - yy) + min(xx, yy)
        # try:
        #     reward = -math.log(layer/2.0, 2)
        # except:
        #     reward = 100.0
        #print(reward)

        # Constant
        #reward = -1

        # Euclidean distance
        reward = -math.sqrt(pow(xsi - xdi, 2) + pow(ysi - ydi, 2))

        #reward = -(abs(xsi - xdi) + abs(ysi - ydi))

        # If doesn't move, then give punishment
        if base_action[0] == 0 and base_action[1] == 0:
            reward -= 15

        if xsi == xdi and ysi == ydi:
            done = True
            reward = 0
        else:
            done = False

        return s_, reward, done, None

    def s_index(self):
        x, y = self.s_position()
        return int(x), int(y)

    def d_index(self):
        x, y = self.d_position()
        return int(x), int(y)

    # Source position
    def s_position(self):
        s = self.canvas.coords(self.rect)
        x = (s[0] + s[2]) / 2.0 / UNIT
        y = (s[1] + s[3]) / 2.0 / UNIT
        return x, y

    # Destination position
    def d_position(self):
        s = self.canvas.coords(self.oval)
        x = (s[0] + s[2]) / 2.0 / UNIT
        y = (s[1] + s[3]) / 2.0 / UNIT
        return x, y

    def render(self):
        #time.sleep(0.1)
        self.update()

    @staticmethod
    def generate_env(des, ori):
        # Initialize Matlab engine
        print("Starting Matlab...")
        eng = matlab.engine.start_matlab()
        print("Matlab engine started.")

        print("Generating environment...")
        GCC = []
        # Fixed destination
        #xd, yd = 0.5 + des[0], 0.5 + des[1]

        # Varying destination
        for di in range(MAZE_W):
            print("%d th column destination." % (di))
            GCC.append([])
            for dj in range(MAZE_H):
                print("%d th row destination." % (dj))
                GCC[di].append([])

                # Skip destinations not in des
                if not [di, dj] in des:
                    continue

                xd, yd = 0.5 + di, 0.5 + dj
                # Varying origin
                for i in range(MAZE_W):
                    GCC[di][dj].append([])
                    for j in range(MAZE_H):

                        xs, ys = 0.5 + i, 0.5 + j

                        # Calculate GCC
                        res = eng.gccGenerator(matlab.double([[xs-0.1, ys-0.1, 1.5], [xs-0.1, ys+0.1, 1.5],
                                                              [xs+0.1, ys-0.1, 1.5], [xs+0.1, ys+0.1, 1.5]]),
                                               maxDis,
                                               matlab.double([float(MAZE_W), float(MAZE_H), 3.0]),
                                               matlab.double([xd, yd, 1.5]))

                        GCC[di][dj][i].append(np.array(res))
                        #print("di = %d, dj = %d, i = %d, j = %d" %(di, dj, i, j))
                        #print(res)
                        #print(GCC[di][dj][i][j])


        # Save GCC
        #with open('env_data_%d_%d.env' %(des[0], des[1]), 'wb') as f:
        with open('env_data_%d_%d.env' %(N, len(des)), 'wb') as f:
            pickle.dump(GCC, f)

        print("Environment established.")

    def load_env(self):
        # Load GCC
        try:
            #with open('env_data_%d_%d.env' %(self.des[0], self.des[1]), 'rb') as f:
            with open('env_data_%d_%d.env' %(N, len(self.des)), 'rb') as f:
                self.GCC = pickle.load(f)
        except:
            self.generate_env(self.des, self.ori)
            with open('env_data_%d_%d.env' %(N, len(self.des)), 'rb') as f:
                self.GCC = pickle.load(f)


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
