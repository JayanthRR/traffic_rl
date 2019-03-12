import matplotlib
from matplotlib import pyplot as plt
import glob
import pickle
import itertools
import numpy as np
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

# folders = glob.glob("/home/jayanth/2019/traffic_rl/new_logs/sp4/2019*/")
rootdir = "/home/cc/traffic_rl/"
pltdir = rootdir + "pltlogs/"

exp_folders = glob.glob(rootdir + "logs/*/")

# cross_loss[sparsity][variance][size][ind]

for exp in exp_folders:
    siz_folders = glob.glob(exp + "*/")

    for siz_folder in siz_folders:
        var_folders = glob.glob(siz_folder + "*/")

        cross_loss = dict()

        for folder in var_folders:
            print(folder)
            marker = itertools.cycle(('X', '+', 'o', '.', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9'))

            files = glob.glob(folder + "*.p")
            files = [f[len(folder):] for f in files]

            with open(folder + "config.p", "rb") as f:
                config = pickle.load(f)

            print("config size", config["size"])
            test_config = config["test"]
            print(test_config.keys())

            sparsity = config["sparsity"]
            variance = config["noise variance"]
            costfn = config["costfn"]
            # costfn = 0
            size = config["size"]
            if variance not in cross_loss.keys():
                cross_loss[variance] = dict().fromkeys(test_config.keys())

            if "logs.p" in files:
                with open(folder + "logs.p", "rb") as f:
                    logdict = pickle.load(f)

                for ind in test_config.keys():
                    rewards = logdict[ind]["rewards"]

                    plt.plot(list(range(len(rewards))), [-reward for reward in rewards], marker=next(marker),
                             label="costs: " + test_config[ind]["algorithm"])
                    plt.legend()
                    plt.xlabel("trials")
                    plt.ylabel("costs")
                    plt.title("Spars: " + str(sparsity) + ", Var: " + str(variance) + ", cfn: " + str(costfn) +
                              ", Network size: " + str(config["size"]))
                    # cross_loss[config["size"]][ind]["avg rewards"] = -np.mean(rewards)
                    cross_loss[variance][ind]["avg rewards"] = -np.mean(rewards)

                plt.savefig(folder + "Test results rewards.pdf", transparent=True, bbox_inches='tight',
                            pad_inches=0)
                plt.close()

                for ind in test_config.keys():
                    paths = logdict[ind]["path"]

                    plt.plot(list(range(len(paths))), [len(path) for path in paths], marker=next(marker),
                             label="num steps: " + test_config[ind]["algorithm"])
                    plt.legend()
                    plt.xlabel("trials")
                    plt.ylabel("number of steps")
                    plt.title("Spars: " + str(sparsity) + ", Var: " + str(variance) + ", cfn: " + str(costfn) +
                              ", Network size: " + str(config["size"]))
                    # cross_loss[config["size"]][ind]["avg steps"] = np.mean([len(path) for path in paths])
                    cross_loss[variance][ind]["avg steps"] = np.mean([len(path) for path in paths])

                plt.savefig(folder + "Test results num steps.pdf", transparent=True, bbox_inches='tight',
                            pad_inches=0)
                plt.close()

                if cross_loss[variance][0]["avg steps"] >= 4 * size:
                    del cross_loss[sparsity][variance][costfn][size]

                # Training logs

                marker = itertools.cycle(('X', '+', 'o', '.', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9'))

                with open(folder + "_qlearning_log.p", "rb") as f:
                    qldict = pickle.load(f)

                num_episodes = config["num episodes"]

                plt.semilogy(list(range(num_episodes))[:], [-x for x in qldict[1][:]], label="Q learning cost")

                plt.ylabel("cost")
                plt.xlabel("episodes")
                plt.title("Spars: " + str(sparsity) + ", Var: " + str(variance) + ", cfn: " + str(costfn) +
                          ", Network size: " + str(config["size"]))
                plt.savefig(folder + "Q Learning training rewards.pdf", transparent=True, bbox_inches='tight', pad_inches=0)
                plt.close()

                plt.semilogy(list(range(num_episodes))[:], [len(path) for path in qldict[0][:]],
                             label="Q learning steps per episodes")

                plt.ylabel("num steps")
                plt.xlabel("episodes")
                plt.title("Spars: " + str(sparsity) + ", Var: " + str(variance) + ", cfn: " + str(costfn) +
                          ", Network size: " + str(config["size"]))
                plt.savefig(folder + "Q Learning training num steps.pdf", transparent=True, bbox_inches='tight',
                            pad_inches=0)
                plt.close()

            else:
                del cross_loss[variance]

        for ind in test_config.keys():

            if ind == 0:
                plt.plot(sorted(cross_loss.keys()),
                         [cross_loss[var][ind]["avg rewards"] for var in
                          sorted(cross_loss.keys())],
                         label="Q learning policy", marker=next(marker))
            else:
                plt.plot(sorted(cross_loss.keys()),
                         [cross_loss[var][ind]["avg rewards"] for var in
                          sorted(cross_loss.keys())],
                         label="const Dijkstra policy", marker=next(marker))
            plt.legend()
            plt.xlabel("size of network")
            plt.ylabel("cost")
            plt.title("Spars: " + str(sparsity) + ", size: " + str(size)+"_cfn_"+str(costfn))
            plt.savefig(siz_folder+"Sp_"+str(sparsity)+"_size_"+str(size)+"_cfn_"+str(costfn)+"_avg rewards.pdf",
                        transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

            for ind in test_config.keys():

                if ind==0:
                    plt.plot(sorted(cross_loss.keys()),
                             [cross_loss[var][ind]["avg steps"] for var in
                              sorted(cross_loss.keys())],
                             label="Q learning policy", marker=next(marker))
                else:
                    plt.plot(sorted(cross_loss.keys()),
                             [cross_loss[var][ind]["avg steps"] for var in
                              sorted(cross_loss.keys())],
                             label="const Dijkstra policy", marker=next(marker))

            plt.legend()
            plt.xlabel("size of network")
            plt.ylabel("num steps")
            plt.title("Spars: " + str(sparsity) + ", size: " + str(size)+"_cfn_"+str(costfn))
            plt.savefig(siz_folder+"Sp_"+str(sparsity)+"_size_"+str(size)+"_cfn_"+str(costfn)+"_avg steps.pdf",
                        transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()
