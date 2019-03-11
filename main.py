from rl import *
from multiprocessing import Pool
import gc
import pickle
import datetime
import os
from tqdm import tqdm

marker_1 = itertools.cycle(('X', '+', 'o', '.', '*', '-', '1', '2', '3', '4', '5'))

config = {"num episodes": 5000,
          "epsilon": 0.2,
          "gamma": 1,
          "sparsity": 0.1,
          "noise variance": 0.01,
          "noise amplitude": 1,
          "noise mean": 0,
          "basis expansion": 0,
          "costfn": 0,
          "learning rate": 0.1,
          "exploration decay": 0.99,
          "size": 160,
          "algorithm": "qlearning",
          "lookahead": [0],
          "num trials": 100,
          "test": {
                    # 0: {"algorithm": "dijkstra",
                    #    "lookahead": 0
                    #    },
                   # 1: {"algorithm": "dijkstra",
                   #     "lookahead": 5
                   #     },
                   # 2: {"algorithm": "sarsa"},
                   0: {"algorithm": "qlearning"},
                   1: {"algorithm": "const_dijkstra",
                       "lookahead": 0
                       }
                   }
          }


def execute_training(args):

    env, config, algo, folder = args

    num_episodes = config["num episodes"]
    epsilon = config["epsilon"]
    gamma = config["gamma"]
    learning_rate = config["learning rate"]
    exploration_decay = config["exploration decay"]

    agent = TrafficAgent(env, learning_rate, epsilon, exploration_decay=exploration_decay)
    seed = np.random.get_state()

    if algo == "qlearning":
        agent, training_rewards, total_rewards = train_agent(agent, num_episodes, gamma, softmax=False, expected=False,
                                                             qlearning=True)
    else:
        agent, training_rewards, total_rewards = train_agent(agent, num_episodes, gamma, softmax=False, expected=False,
                                                             qlearning=False)

    agent_file = folder + algo + ".p"
    log_file = folder + "_" + algo + "_log.p"
    rand_seed = folder + "_seed_.p"

    with open(agent_file, "wb") as f:
        pickle.dump(agent, f)
    with open(log_file, "wb") as f:
        pickle.dump([training_rewards, total_rewards], f)
    with open(rand_seed, "wb") as f:
        pickle.dump(seed, f)

    del agent, env
    gc.collect()


def genAfortrain(config):

    size = config["size"]
    sparsity = config["sparsity"]

    transition_matrix = generate_column_stochastic_matrix(size, sparsity)
    source = np.random.randint(size)
    destination, min_len = longest_destination(transition_matrix, source, min_len=int(size / 4))

    print(source, destination, min_len)

    while True:
        if min_len is None:
            transition_matrix = generate_column_stochastic_matrix(size, sparsity)
            source = np.random.randint(size)
            destination, min_len = longest_destination(transition_matrix, source, min_len=int(size / 4))
        elif min_len <= 2:
            transition_matrix = generate_column_stochastic_matrix(size, sparsity)
            source = np.random.randint(size)
            destination, min_len = longest_destination(transition_matrix, source, min_len=int(size / 4))
        else:
            break

    return transition_matrix, source, destination, min_len


def evaluate(args):

    env, agent, W, config, config_id = args
    algo = config[config_id]["algorithm"]

    if algo in ["qlearning", "sarsa"]:
        agent[algo].env = env
        reward, _, path = evaluate_rl_policy(agent[algo], W)
    elif algo in ["const_dijkstra"]:
        lookahead = config[config_id]["lookahead"]
        reward, _, path = evaluate_policies(env, W, algo, lookahead, const_flag=True)
    else:
        lookahead = config[config_id]["lookahead"]
        reward, _, path = evaluate_policies(env, W, algo, lookahead)

    logs = dict()
    logs["rewards"] = reward
    logs["path"] = path
    gc.collect()
    del env, agent, W
    return logs


def test(config, folder):

    num_trials = config["num trials"]
    size = config["size"]
    noise_var = config["noise variance"]
    noise_mean = config["noise mean"]
    noise_amp = config["noise amplitude"]

    time_steps = 5 * size
    test_config = config["test"]
    # print(test_config)
    logdict = dict().fromkeys(test_config.keys())
    for ind in test_config.keys():
        logdict[ind] = dict()
        logdict[ind]["rewards"] = []
        logdict[ind]["path"] = []

    with open(folder+"env.p", "rb") as f:
        env = pickle.load(f)

    agent = dict()
    algo = config["algorithm"]
    agent_file = folder + algo + ".p"
    with open(agent_file, "rb") as f:
        agent[algo] = pickle.load(f)

    test_seed = np.random.get_state()
    with open(folder+"_testseed.p", "wb") as f:
        pickle.dump(test_seed, f)

    for trial in tqdm(range(num_trials)):
        W = []
        for i in range(time_steps):
            W.append(gaussian(size, mean=noise_mean, sigma=noise_var))

        env.reset()
        logs={}

        for k in test_config.keys():
            logs[k] = evaluate([copy.deepcopy(env), agent, W, test_config, k])

        for ind in test_config.keys():
            logdict[ind]["rewards"].append(logs[ind]["rewards"])
            logdict[ind]["path"].append(logs[ind]["path"])
        gc.collect()

    log_file = folder + "logs.p"
    with open(log_file, "wb") as f:
        pickle.dump(logdict, f)


def run(config, A, source, destination, costdict, folder):

    folder += datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(folder+"config.p", "wb") as f:
        pickle.dump(config, f)

    size = config["size"]
    noise_var = config["noise variance"]
    noise_mean = config["noise mean"]
    noise_amp = config["noise amplitude"]
    be = config["basis expansion"]

    x_init = np.random.rand(size)
    x_init = x_init / x_init.sum()

    env = TrafficEnv(A, x_init, source, destination, costdict, be,
                     noise_mean=noise_mean, noise_var=noise_var, noise_amp=noise_amp)

    env_file = folder + "env.p"
    with open(env_file, "wb") as f:
        pickle.dump(env, f)

    args = (copy.deepcopy(env), config, config["algorithm"], folder)
    execute_training(args)
    print("completed training ...")

    test(config, folder)
    print("completed testing ...")


def gencostfn(cfn, size):

    if cfn == 1:
        costdict = [(0, 1) for _ in range(size)]
    elif cfn == 2:
        cmax = np.random.uniform(0.5, 1.5, size=(size,))
        costdict = [(0, cx) for cx in cmax]
    else:
        cmin = np.random.uniform(1e-5, 1e-4, size=(size,))
        cmax = np.random.uniform(0.5, 1.5, size=(size,))
        costdict = [(cm, cx) for [cm, cx] in zip(cmin, cmax)]

    return costdict


if __name__ == "__main__":

    config["sparsity"] = 0.05
    # for nv in tqdm([0.01, 0.1]):
    #
    #     config["noise variance"] = nv
    #
    #     for size in [180]:
    #         config["size"]=size
    #         run(config)

    folder = "logs/"

    for be in [0, 1, 2]:
        config["basis expansion"] = be

        for cfn in [1, 2, 3]:
            config["costfn"] = cfn
            folder = "logs/exp_" + str(3*be+cfn) + "/"

            np.random.seed(1729)

            for siz in [150, 200, 250]:
                config["size"] = siz
                folder += str(siz) + "/"

                A, source, destination, _ = genAfortrain(config)
                costdict = gencostfn(cfn, siz)

                for var in tqdm([0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.12, 0.14, 0.16]):
                    config["noise variance"] = var
                    folder += str(var) + "_"

                    run(config, A, source, destination, costdict, folder)

