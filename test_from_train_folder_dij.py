import glob
import pickle
from main import test
import os
from graphs import const_dijkstra_policy


folders = glob.glob("/home/cc/traffic_rl/logs/2018*/")
# folders = glob.glob("/home/jayanth/thesis/traffic_rl/logs/2018*/")

for folder in folders:

    files = glob.glob(folder + "*.p")
    files = [f[len(folder):] for f in files]
    print(files)

    if "logs.p" in files:
        with open(folder + "config.p", "rb") as f:
            config = pickle.load(f)

        with open(folder+"env.p", "rb") as f:
            env = pickle.load(f)


