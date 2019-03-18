import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import glob
import pickle
import itertools
import numpy as np
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}


def plot(folder_name=None):
    # folders = glob.glob("/home/jayanth/2019/traffic_rl/new_logs/sp4/2019*/")

    # rootdir = "/home/jayanth/2019/traffic_rl/"
    rootdir = "/home/cc/traffic_rl/"
    pltdir = rootdir + "pltlogs/"

    if not folder_name:
        exp_folders = glob.glob(rootdir + "logs/2019-03-17-12-19-49/*/")
    else:
        exp_folders = glob.glob(rootdir + folder_name + "*/")

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
                    cross_loss[variance] = dict()
                for ind in test_config.keys():
                    cross_loss[variance][ind] = dict()

                if "logs.p" in files:
                    with open(folder + "logs.p", "rb") as f:
                        logdict = pickle.load(f)
                    marker = itertools.cycle(('X', '+', 'o', '.', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9'))

                    for ind in test_config.keys():
                        rewards = logdict[ind]["rewards"]

                        plt.plot(list(range(len(rewards))), [-reward for reward in rewards], marker=next(marker),
                                 label="costs: " + test_config[ind]["algorithm"])
                        plt.legend()
                        plt.xlabel("trials")
                        plt.ylabel("costs")
                        plt.title("Var: " + str(variance) + ", cfn: " + str(costfn) +
                                  ", Network size: " + str(config["size"]))
                        # cross_loss[config["size"]][ind]["avg rewards"] = -np.mean(rewards)
                        cross_loss[variance][ind]["avg rewards"] = -np.mean(rewards)
                        cross_loss[variance][ind]["std rewards"] = -np.std(rewards)
                        cross_loss[variance][ind]["median rewards"] = -np.median(rewards)
                        cross_loss[variance][ind]["rewards"] = [-reward for reward in rewards]

                    plt.savefig(folder + "Test results rewards.png", transparent=True, bbox_inches='tight',
                                pad_inches=0)
                    plt.close()
                    marker = itertools.cycle(('X', '+', 'o', '.', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9'))

                    for ind in test_config.keys():
                        paths = logdict[ind]["path"]

                        plt.plot(list(range(len(paths))), [len(path) for path in paths], marker=next(marker),
                                 label="num steps: " + test_config[ind]["algorithm"])
                        plt.legend()
                        plt.xlabel("trials")
                        plt.ylabel("number of steps")
                        plt.title("Var: " + str(variance) + ", cfn: " + str(costfn) +
                                  ", Network size: " + str(config["size"]))
                        # cross_loss[config["size"]][ind]["avg steps"] = np.mean([len(path) for path in paths])
                        cross_loss[variance][ind]["avg steps"] = np.mean([len(path) for path in paths])
                        cross_loss[variance][ind]["std steps"] = np.std([len(path) for path in paths])
                        cross_loss[variance][ind]["median steps"] = np.median([len(path) for path in paths])
                        cross_loss[variance][ind]["paths"] = paths

                    plt.savefig(folder + "Test results num steps.png", transparent=True, bbox_inches='tight',
                                pad_inches=0)
                    plt.close()

                    if cross_loss[variance][0]["avg steps"] >= 4 * size:
                        del cross_loss[variance]

                    # Training logs

                    marker = itertools.cycle(('X', '+', 'o', '.', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9'))

                    with open(folder + "_qlearning_log.p", "rb") as f:
                        qldict = pickle.load(f)

                    num_episodes = config["num episodes"]

                    plt.semilogy(list(range(num_episodes))[:3000], [-x for x in qldict[1][:3000]], label="Q learning cost")

                    plt.ylabel("cost")
                    plt.xlabel("episodes")
                    plt.title("Q learning training")
                    plt.savefig(folder + "Var: " + str(variance) + ", cfn: " + str(costfn) + ", Network size: " +
                                str(config["size"]) + "Q Learning training cost.png", transparent=True,
                                bbox_inches='tight', pad_inches=0)
                    plt.close()

                    plt.semilogy(list(range(num_episodes))[:3000], [len(path) for path in qldict[0][:3000]],
                                 label="Q learning steps per episodes")

                    plt.ylabel("number of steps")
                    plt.xlabel("episodes")
                    plt.title("Q learning training")
                    plt.savefig(folder + "Var: " + str(variance) + ", cfn: " + str(costfn) + ", Network size: " +
                                str(config["size"]) + "Q Learning training num steps.png", transparent=True,
                                bbox_inches='tight', pad_inches=0)
                    plt.close()

                else:
                    del cross_loss[variance]

            marker = itertools.cycle(('X', '+', 'o', '.', '*', '1', '2'))
            linestyles = itertools.cycle(('-', '--', '-.', ':',))
            for ind in test_config.keys():

                if test_config[ind]["algorithm"] == "qlearning":
                    plt.errorbar(sorted(cross_loss.keys()),
                                 [cross_loss[var][ind]["avg rewards"] for var in sorted(cross_loss.keys())],
                                 label="Q learning policy", marker=next(marker), linestyle=next(linestyles))
                elif test_config[ind]["algorithm"] == "const_dijkstra":
                    # if test_config[ind]["lookahead"] in [1, 5]:
                    #     continue

                    plt.errorbar(sorted(cross_loss.keys()),
                                 [cross_loss[var][ind]["avg rewards"] for var in sorted(cross_loss.keys())],
                                 label="Dijkstra policy, lookahead: "+ str(test_config[ind]["lookahead"]),
                                 marker=next(marker), linestyle=next(linestyles))

                elif test_config[ind]["algorithm"] == "expected_dijkstra":
                    plt.errorbar(sorted(cross_loss.keys()),
                                 [cross_loss[var][ind]["avg rewards"] for var in sorted(cross_loss.keys())],
                                 label="expected Dijkstra policy",
                                 marker=next(marker), linestyle=next(linestyles))
            plt.legend()
            plt.xlabel("noise variance")
            plt.ylabel("cost")
            plt.title("Average Cost")
            plt.savefig(siz_folder+"_size_"+str(size)+"_cfn_"+str(costfn)+"_avg_rewards.png",
                        transparent=True, bbox_inches='tight', pad_inches=0)
            plt.savefig(siz_folder+"_size_"+str(size)+"_cfn_"+str(costfn)+"_avg_rewards.pdf",
                        transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()
            """
            marker = itertools.cycle(('X', '+', 'o', '.', '*', '1', '2'))
            for ind in test_config.keys():

                if test_config[ind]["algorithm"] == "qlearning":
                    plt.errorbar(sorted(cross_loss.keys()),
                                 [cross_loss[var][ind]["avg steps"] for var in sorted(cross_loss.keys())],
                                 label="Q learning policy", marker=next(marker))
                elif test_config[ind]["algorithm"] == "const_dijkstra":
                    plt.errorbar(sorted(cross_loss.keys()),
                                 [cross_loss[var][ind]["avg steps"] for var in sorted(cross_loss.keys())],
                                 label="const Dijkstra policy_"+str(test_config[ind]["lookahead"]),
                                 marker=next(marker))

                elif test_config[ind]["algorithm"] == "expected_dijkstra":
                    plt.errorbar(sorted(cross_loss.keys()),
                                 [cross_loss[var][ind]["avg steps"] for var in sorted(cross_loss.keys())],
                                 label="expected Dijkstra policy",
                                 marker=next(marker))

            plt.legend()
            plt.xlabel("noise variance")
            plt.ylabel("num steps")
            plt.title("size: " + str(size)+"_cfn_"+str(costfn))
            plt.savefig(siz_folder+"_size_"+str(size)+"_cfn_"+str(costfn)+"_avg steps.png",
                        transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

            marker = itertools.cycle(('X', '+', 'o', '.', '*', '1', '2'))
            for ind in test_config.keys():

                if test_config[ind]["algorithm"] == "qlearning":
                    plt.errorbar(sorted(cross_loss.keys()),
                                 [cross_loss[var][ind]["median rewards"] for var in sorted(cross_loss.keys())],
                                 label="Q learning policy", marker=next(marker))
                elif test_config[ind]["algorithm"] == "const_dijkstra":
                    plt.errorbar(sorted(cross_loss.keys()),
                                 [cross_loss[var][ind]["median rewards"] for var in sorted(cross_loss.keys())],
                                 label="const Dijkstra policy_"+ str(test_config[ind]["lookahead"]),
                                 marker=next(marker))

                elif test_config[ind]["algorithm"] == "expected_dijkstra":
                    plt.errorbar(sorted(cross_loss.keys()),
                                 [cross_loss[var][ind]["median rewards"] for var in sorted(cross_loss.keys())],
                                 label="expected Dijkstra policy",
                                 marker=next(marker))
            plt.legend()
            plt.xlabel("noise variance")
            plt.ylabel("cost")
            plt.title("size: " + str(size)+"_cfn_"+str(costfn))
            plt.savefig(siz_folder+"_size_"+str(size)+"_cfn_"+str(costfn)+"_median rewards.png",
                        transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

            for var in cross_loss.keys():
                keys = [ind for ind in test_config.keys()]
                labels = [test_config[ind]["algorithm"] for ind in keys]
                hist_costs = [cross_loss[var][ind]["rewards"] for ind in keys]
                plt.hist(hist_costs, 10, histtype="bar", label=labels)
                plt.legend()
                plt.xlabel("cost bins")
                plt.ylabel("counts")
                plt.title("Histogram of costs: var_" + str(var) + "_size_" + str(size)+"_cfn_"+str(costfn))
                plt.savefig(siz_folder+"_var_" + str(var) + "_size_" + str(size)+
                            "_cfn_"+str(costfn)+"_hist.png")
                plt.close()

            for var in cross_loss.keys():
                keys = sorted([ind for ind in test_config.keys()])
                labels = [test_config[ind]["algorithm"] for ind in keys]
                pathdict = {}
                for ind in keys:
                    for trial in range(len(cross_loss[var][ind]["paths"])):
                        pstr = '_'.join([str(p) for p in cross_loss[var][ind]["paths"][trial]])
                        if pstr not in pathdict.keys():
                            pathdict[pstr] = dict.fromkeys(test_config.keys(), 0)

                        pathdict[pstr][ind] += 1

                pstrs = list(pathdict.keys())
                index = np.arange(len(pstrs))
                pathstable = np.zeros((len(pstrs), len(keys)))
                for pind in range(len(pstrs)):
                    for ind in keys:
                        pathstable[pind][ind] = pathdict[pstrs[pind]][ind]

                bar_width = 0.15
                for ind in keys:
                    plt.bar(index + ind*bar_width, pathstable[:, ind], bar_width,
                            label=test_config[ind]["algorithm"])

                plt.legend()
                # plt.xticks(None)
                plt.xlabel("paths")
                plt.ylabel("counts")
                plt.title("Histogram of paths: var_" + str(var) + "_size_" + str(size)+"_cfn_"+str(costfn))
                plt.savefig(siz_folder+"_var_" + str(var) + "_size_" + str(size)+
                            "_cfn_"+str(costfn)+"_path_hist.png")
                plt.close()

            """


if __name__=="__main__":

    plot()
