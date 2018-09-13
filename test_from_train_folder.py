import glob
import pickle
from main import test
import os


# folders = glob.glob("/home/cc/traffic_rl/logs/2018*/")
folders = glob.glob("/home/jayanth/thesis/traffic_rl/logs/2018-09-05-19-09-49/")

for folder in folders:

    files = glob.glob(folder + "*.p")
    files = [f[len(folder):] for f in files]
    print(files)

    if not "logs.p" in files:
        if ("qlearning.p" in files):

            with open(folder+"config.p", "rb") as f:
                config=pickle.load(f)

            config["test"][2] = {"algorithm": "const_dijkstra",
                                 "lookahead": 0
                                 }
            with open(folder+"config.p", "wb") as f:
                pickle.dump(config, f)

            test(config, folder)
        else:
            print("testing not done")
            for file in files:
                os.remove(folder + file)
            os.rmdir(folder)

    else:
        print("training and testing done")