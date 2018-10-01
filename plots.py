import matplotlib
from matplotlib import pyplot as plt
import glob
import pickle
import itertools
import numpy as np
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}


# matplotlib.rc('font', **font)

# folders = glob.glob("/home/jayanth/thesis/traffic_rl/logs/2018-09-05-19-09-49/")
folders = glob.glob("/home/jayanth/thesis/traffic_rl/logs_v2/2018*/")
cross_loss = dict()

for folder in folders:
    print(folder)
    marker = itertools.cycle(('X', '+', 'o', '.', '*', '1', '2', '3', '4','5','6','7','8','9'))

    files = glob.glob(folder + "*.p")
    files = [f[len(folder):] for f in files]

    with open(folder + "config.p", "rb") as f:
        config = pickle.load(f)

    if config["size"]==70:
        # some error with data collection. Did not record const_dijkstra test for this log
        continue

    print("config size", config["size"])
    test_config = config["test"]
    print(test_config.keys())
    cross_loss[config["size"]] = dict()
    cross_loss[config["size"]][0] = dict()
    cross_loss[config["size"]][1] = dict()
    cross_loss[config["size"]][2] = dict()

    #### Testing plots

    if "logs.p" in files:
        with open(folder + "logs.p", "rb") as f:
            logdict = pickle.load(f)

        for ind in test_config.keys():
        # for ind in [0, 1]:
            rewards = logdict[ind]["rewards"]
            # if test_config[ind]["algorithm"]=="dijkstra":
            #     plt.plot(list(range(len(rewards))), rewards, marker=next(marker),
            #              label="rewards: " + test_config[ind]["algorithm"]+" lookahead: "+str(test_config[ind]["lookahead"]))
            # else:
            plt.plot(list(range(len(rewards))), [-reward for reward in rewards], marker=next(marker), label="costs: " + test_config[ind]["algorithm"])
            plt.legend()
            plt.xlabel("trials")
            plt.ylabel("costs")
            plt.title("Network size: "+str(config["size"]))
            cross_loss[config["size"]][ind]["avg rewards"] = -np.mean(rewards)

        plt.savefig(folder+"Test results rewards.pdf", transparent=True, bbox_inches='tight',
                    pad_inches=0)
        plt.close()
        for ind in test_config.keys():
        # for ind in [0, 1]:
            paths = logdict[ind]["path"]
            print([len(path) for path in paths])
            # if test_config[ind]["algorithm"]=="dijkstra":
            #     plt.plot(list(range(len(paths))), [len(path) for path in paths], marker=next(marker),
            #              label="num steps: " + test_config[ind]["algorithm"]+" lookahead: "+str(test_config[ind]["lookahead"]))
            # else:
            plt.plot(list(range(len(paths))), [len(path) for path in paths], marker=next(marker),
                     label="num steps: " + test_config[ind]["algorithm"])
            plt.legend()
            plt.xlabel("trials")
            plt.ylabel("number of steps")
            plt.title("Network size: "+str(config["size"]))
            cross_loss[config["size"]][ind]["avg steps"] = np.mean([len(path) for path in paths])

        plt.savefig(folder + "Test results num steps.pdf", transparent=True, bbox_inches='tight',
                    pad_inches=0)
        plt.close()

    #### Training plots
        marker = itertools.cycle(('X', '+', 'o', '.', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9'))

        with open(folder + "_qlearning_log.p", "rb") as f:
            qldict = pickle.load(f)
        # with open(folder + "_sarsa_log.p", "rb") as f:
        #     srdict = pickle.load(f)
        num_episodes=config["num episodes"]

        plt.semilogy(list(range(num_episodes))[:], [-x for x in qldict[1][:]], label="Q learning cost")
        # plt.legend()
        plt.ylabel("cost")
        plt.xlabel("episodes")
        plt.title("Network size: " + str(config["size"]))
        plt.savefig(folder + "Q Learning training rewards.pdf", transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.semilogy(list(range(num_episodes))[:], [len(path) for path in qldict[0][:]], label="Q learning steps per episodes")
        # plt.legend()
        plt.ylabel("num steps")
        plt.xlabel("episodes")
        plt.title("Network size: " + str(config["size"]))
        plt.savefig(folder + "Q Learning training num steps.pdf", transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()
        # plt.plot(list(range(num_episodes))[::10], srdict[1][::10], label="Sarsa learning rewards",marker=next(marker))
        # plt.plot(list(range(num_episodes))[::10], [len(path) for path in srdict[0][::10]], label="Sarsa learning steps per episodes", marker=next(marker))
        # plt.legend()
        # plt.xlabel("episodes")
        # plt.title("Network size: " + str(config["size"]))
        # plt.savefig(folder + "Sarsa Learning training.pdf", transparent=True, bbox_inches='tight')
        # plt.close()

    else:
        print("No log found")


plt.close()

for ind in [0,1,2]:
    if ind==0:
        plt.plot(sorted(cross_loss.keys()), [cross_loss[size][ind]["avg rewards"] for size in sorted(cross_loss.keys())], label="dijkstra policy", marker=next(marker))
    elif ind==1:
        plt.plot(sorted(cross_loss.keys()), [cross_loss[size][ind]["avg rewards"] for size in sorted(cross_loss.keys())], label="Q learning policy", marker=next(marker))
    else:
        plt.plot(sorted(cross_loss.keys()), [cross_loss[size][ind]["avg rewards"] for size in sorted(cross_loss.keys())], label="const Dijkstra policy", marker=next(marker))


plt.legend()
plt.xlabel("size of network")
plt.ylabel("cost")
plt.title("Average cost accrued")
plt.savefig("avg rewards.pdf", transparent=True, bbox_inches='tight', pad_inches=0)

plt.close()
for ind in [0,1,2]:
    if ind==0:
        plt.plot(sorted(cross_loss.keys()), [cross_loss[size][ind]["avg steps"] for size in sorted(cross_loss.keys())], label="dijkstra policy", marker=next(marker))
    elif ind==1:
        plt.plot(sorted(cross_loss.keys()), [cross_loss[size][ind]["avg steps"] for size in sorted(cross_loss.keys())], label="Q learning policy", marker=next(marker))
    else:
        plt.plot(sorted(cross_loss.keys()), [cross_loss[size][ind]["avg steps"] for size in sorted(cross_loss.keys())], label="const Dijkstra policy", marker=next(marker))

plt.legend()
plt.xlabel("size of network")
plt.ylabel("num steps")
plt.title("Average steps taken")
plt.savefig("avg steps.pdf", transparent=True, bbox_inches='tight', pad_inches=0)



