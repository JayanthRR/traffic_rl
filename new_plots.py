import matplotlib
from matplotlib import pyplot as plt
import glob
import pickle
import itertools
import numpy as np
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

# folders = glob.glob("/home/jayanth/thesis/traffic_rl/logs/2018-09-05-19-09-49/")
folders = glob.glob("/home/jayanth/2019/traffic_rl/new_logs/sp3/2019*/")
cross_loss = dict()
# cross_loss[sparsity][variance][size][ind]

for folder in folders:
    print(folder)
    marker = itertools.cycle(('X', '+', 'o', '.', '*', '1', '2', '3', '4','5','6','7','8','9'))

    files = glob.glob(folder + "*.p")
    files = [f[len(folder):] for f in files]

    with open(folder + "config.p", "rb") as f:
        config = pickle.load(f)

    print("config size", config["size"])
    test_config = config["test"]
    print(test_config.keys())

    sparsity = config["sparsity"]
    variance = config["noise variance"]
    size = config["size"]

    if sparsity not in cross_loss.keys():
        cross_loss[sparsity] = dict()

    if variance not in cross_loss[sparsity].keys():
        cross_loss[sparsity][variance] = dict()

    cross_loss[sparsity][variance][size] = dict()

    cross_loss[sparsity][variance][size][0] = dict()
    cross_loss[sparsity][variance][size][1] = dict()
    cross_loss[sparsity][variance][size][2] = dict()

    if "logs.p" in files:
        with open(folder + "logs.p", "rb") as f:
            logdict = pickle.load(f)

        # Testing plots

        for ind in test_config.keys():

            rewards = logdict[ind]["rewards"]

            plt.plot(list(range(len(rewards))), [-reward for reward in rewards], marker=next(marker), label="costs: " + test_config[ind]["algorithm"])
            plt.legend()
            plt.xlabel("trials")
            plt.ylabel("costs")
            plt.title("Spars: " + str(sparsity) + ", Var: " + str(variance) +
                      ", Network size: "+str(config["size"]))
            # cross_loss[config["size"]][ind]["avg rewards"] = -np.mean(rewards)
            cross_loss[sparsity][variance][size][ind]["avg rewards"] = -np.mean(rewards)

        plt.savefig(folder+"Test results rewards.pdf", transparent=True, bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        for ind in test_config.keys():

            paths = logdict[ind]["path"]
            # print([len(path) for path in paths])
            plt.plot(list(range(len(paths))), [len(path) for path in paths], marker=next(marker),
                     label="num steps: " + test_config[ind]["algorithm"])
            plt.legend()
            plt.xlabel("trials")
            plt.ylabel("number of steps")
            plt.title("Spars: " + str(sparsity) + ", Var: " + str(variance) +
                      ", Network size: "+str(config["size"]))
            # cross_loss[config["size"]][ind]["avg steps"] = np.mean([len(path) for path in paths])
            cross_loss[sparsity][variance][size][ind]["avg steps"] = np.mean([len(path) for path in paths])

        plt.savefig(folder + "Test results num steps.pdf", transparent=True, bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        # Training plots
        marker = itertools.cycle(('X', '+', 'o', '.', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9'))

        with open(folder + "_qlearning_log.p", "rb") as f:
            qldict = pickle.load(f)

        num_episodes=config["num episodes"]

        plt.semilogy(list(range(num_episodes))[:], [-x for x in qldict[1][:]], label="Q learning cost")

        plt.ylabel("cost")
        plt.xlabel("episodes")
        plt.title("Spars: " + str(sparsity) + ", Var: " + str(variance) +
                  ", Network size: " + str(config["size"]))
        plt.savefig(folder + "Q Learning training rewards.pdf", transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.semilogy(list(range(num_episodes))[:], [len(path) for path in qldict[0][:]], label="Q learning steps per episodes")

        plt.ylabel("num steps")
        plt.xlabel("episodes")
        plt.title("Spars: " + str(sparsity) + ", Var: " + str(variance) +
                  ", Network size: " + str(config["size"]))
        plt.savefig(folder + "Q Learning training num steps.pdf", transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

    else:
        del cross_loss[sparsity][variance][size]
        print("No log found")


for spars in cross_loss.keys():

    for var in cross_loss[spars].keys():
        marker = itertools.cycle(('X', '+', 'o', '.', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9'))

        for ind in [0, 1, 2]:
            if ind == 0:
                plt.plot(sorted(cross_loss[spars][var].keys()),
                         [cross_loss[spars][var][size][ind]["avg rewards"] for size in sorted(cross_loss[spars][var].keys())],
                         label="dijkstra policy", marker=next(marker))
            elif ind == 1:
                plt.plot(sorted(cross_loss[spars][var].keys()),
                         [cross_loss[spars][var][size][ind]["avg rewards"] for size in sorted(cross_loss[spars][var].keys())],
                         label="Q learning policy", marker=next(marker))
            else:
                plt.plot(sorted(cross_loss[spars][var].keys()),
                         [cross_loss[spars][var][size][ind]["avg rewards"] for size in sorted(cross_loss[spars][var].keys())],
                         label="const Dijkstra policy", marker=next(marker))

        plt.legend()
        plt.xlabel("size of network")
        plt.ylabel("cost")
        plt.title("Spars: " + str(spars) + ", Var: " + str(var))
        plt.savefig("new_logs/Sp_"+str(spars)+"_var_"+str(var)+"_avg rewards.pdf",
                    transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

        for ind in [0, 1, 2]:
            if ind==0:
                plt.plot(sorted(cross_loss[spars][var].keys()),
                         [cross_loss[spars][var][size][ind]["avg steps"] for size in sorted(cross_loss[spars][var].keys())],
                         label="dijkstra policy", marker=next(marker))
            elif ind==1:
                plt.plot(sorted(cross_loss[spars][var].keys()),
                         [cross_loss[spars][var][size][ind]["avg steps"] for size in sorted(cross_loss[spars][var].keys())],
                         label="Q learning policy", marker=next(marker))
            else:
                plt.plot(sorted(cross_loss[spars][var].keys()),
                         [cross_loss[spars][var][size][ind]["avg steps"] for size in sorted(cross_loss[spars][var].keys())],
                         label="const Dijkstra policy", marker=next(marker))

        plt.legend()
        plt.xlabel("size of network")
        plt.ylabel("num steps")
        plt.title("Spars: " + str(spars) + ", Var: " + str(var))
        plt.savefig("new_logs/Sp_"+str(spars)+"_var_"+str(var)+"_avg steps.pdf",
                    transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()



