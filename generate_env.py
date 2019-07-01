import matlab.engine
import numpy as np
import threading
import pickle

MAZE_W = 9
MAZE_H = 9

maxDis = 0.2828

def function(di, GCC, snr):
        print("Starting Matlab...")
        eng = matlab.engine.start_matlab()
        print("Matlab engine started.")

        # Varying destination
        for dj in range(MAZE_H):
            #print("%d th row destination." % (dj))
            GCC[di].append([])

            # Skip destinations not in des
            #if not [di, dj] in des:
            #    continue

            xd, yd = 0.5 + di, 0.5 + dj
            # Varying origin
            for i in range(MAZE_W):
                GCC[di][dj].append([])
                for j in range(MAZE_H):
                    xs, ys = 0.5 + i, 0.5 + j

                    # Calculate GCC
                    res = eng.gccGenerator(matlab.double([[xs - 0.1, ys - 0.1, 1.5], [xs - 0.1, ys + 0.1, 1.5],
                                                          [xs + 0.1, ys - 0.1, 1.5], [xs + 0.1, ys + 0.1, 1.5]]),
                                           maxDis,
                                           matlab.double([float(MAZE_W), float(MAZE_H), 3.0]),
                                           matlab.double([xd, yd, 1.5]),
                                           float(snr))

                    GCC[di][dj][i].append(np.array(res))
                    # print("di = %d, dj = %d, i = %d, j = %d" %(di, dj, i, j))
                    # print(res)
                    # print(GCC[di][dj][i][j])

def generate_env():
        # Initialize Matlab engine
        # print("Starting Matlab...")
        # eng = matlab.engine.start_matlab()
        # print("Matlab engine started.")

        # print("Generating environment...")
        GCC = []
        for i in range(MAZE_W):
            GCC.append([])
        # Fixed destination
        #xd, yd = 0.5 + des[0], 0.5 + des[1]

        threads = []
        SNR = 40

        print("Generating %d SNR environment..." %(SNR))

        for i in range(MAZE_W):
            t = threading.Thread(target=function, args=(i, GCC, SNR))
            threads.append(t)
            t.start()
        for i in range(MAZE_W):
            threads[i].join()

        # for di in range(MAZE_W):
        #     for dj in range(MAZE_W):
        #         for i in range(MAZE_W):
        #             for j in range(MAZE_W):
        #                 print("des(%d, %d), ori(%d, %d):\n" %(di, dj, i, j))
        #                 print(GCC[di][dj][i][j])

        # Save GCC
        #with open('env_data_%d_%d.env' %(des[0], des[1]), 'wb') as f:
        with open('data/env_data_%d_%d_SNR_%d.env' %(MAZE_W, MAZE_H, SNR), 'wb') as f:
            pickle.dump(GCC, f)

        print("Environment established.")

generate_env()
#env = Maze()