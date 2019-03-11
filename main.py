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
          "costfn": 0,
          "learning rate": 0.1,
          "exploration decay": 0.99,
          "size": 160,
          "algorithm": ["qlearning"],
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
                   1: {"algorithm": "qlearning"},
                   2: {"algorithm": "const_dijkstra",
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


def train(config, folder):

    size = config["size"]
    noise_var = config["noise variance"]
    noise_mean = config["noise mean"]
    noise_amp = config["noise amplitude"]
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

    print("A generated")

    rl_algo = config["algorithm"]

    cmin = np.random.uniform(0.1, 0.2, size=(size,))
    cmax = np.random.uniform(1, 2, size=(size,))
    costdict = [(cm, cx) for (cm, cx) in zip(cmin, cmax)]

    x_init = np.random.rand(size)
    x_init = x_init / x_init.sum()

    env = TrafficEnv(transition_matrix, x_init, source, destination, costdict,
                     noise_mean=noise_mean, noise_var=noise_var, noise_amp=noise_amp)

    env_file = folder + "env.p"
    with open(env_file, "wb") as f:
        pickle.dump(env, f)
    args = [(copy.deepcopy(env), config, algo, folder) for algo in rl_algo]
    pool = Pool(processes=1)
    pool.map(execute_training, args)
    # for arg in args:
    #     execute_training(arg)


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
    for algo in config["algorithm"]:
        agent_file = folder + algo + ".p"
        with open(agent_file, "rb") as f:
            agent[algo] = pickle.load(f)

    test_seed = np.random.get_state()
    with open(folder+"_testseed.p", "wb") as f:
        pickle.dump(test_seed, f)

    pool = Pool(processes=3)
    for trial in tqdm(range(num_trials)):
        W = []
        for i in range(time_steps):
            W.append(gaussian(size, mean=noise_mean, sigma=noise_var))

        env.reset()

        args = [(copy.deepcopy(env), agent, W, test_config, k) for k in test_config.keys()]
        logs = pool.map(evaluate, args)
        # for arg in args:
        #     evaluate(arg)

        for ind in test_config.keys():
            logdict[ind]["rewards"].append(logs[ind]["rewards"])
            logdict[ind]["path"].append(logs[ind]["path"])
        gc.collect()

    log_file = folder + "logs.p"
    with open(log_file, "wb") as f:
        pickle.dump(logdict, f)


def run(config, A, source, destination, costdict):

    folder = "logs/" + str(config["size"]) + "/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "/"

    size = config["size"]
    noise_var = config["noise variance"]
    noise_mean = config["noise mean"]
    noise_amp = config["noise amplitude"]

    x_init = np.random.rand(size)
    x_init = x_init / x_init.sum()

    env = TrafficEnv(A, x_init, source, destination, costdict,
                     noise_mean=noise_mean, noise_var=noise_var, noise_amp=noise_amp)

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(folder+"config.p", "wb") as f:
        pickle.dump(config, f)

    env_file = folder + "env.p"
    with open(env_file, "wb") as f:
        pickle.dump(env, f)

    args = [(copy.deepcopy(env), config, algo, folder) for algo in config["algorithm"]]
    pool = Pool(processes=1)
    pool.map(execute_training, args)

    print("completed training ...")


if __name__ == "__main__":

    config["sparsity"] = 0.05
    for nv in tqdm([0.01, 0.1]):

        config["noise variance"] = nv

        for size in [100, 200, 300]:
            config["size"]=size
            A, source, destination, _ = genAfortrain(config)

            for costfn in [0, 7]:
                config["costfn"] = costfn
                if costfn == 0:
                    cmin = 0
                    cmax = 1
                    costdict = [(cmin, cmax) for _ in range(size)]
                elif costfn == 1:
                    cmin = 1e-4
                    cmax = 1
                    costdict = [(cmin, cmax) for _ in range(size)]
                elif costfn == 2:
                    cmin = 0
                    cmax = np.random.uniform(1,1.5)
                    costdict = [(cmin, cmax) for _ in range(size)]
                elif costfn == 3:
                    cmin = 0
                    cmax = np.random.uniform(0.5, 1)
                    costdict = [(cmin, cmax) for _ in range(size)]
                elif costfn == 4:
                    cmin = 1e-4
                    cmax = np.random.uniform(1,1.5)
                    costdict = [(cmin, cmax) for _ in range(size)]
                elif costfn == 5:
                    cmin = 1e-4
                    cmax = np.random.uniform(0.5, 1)
                    costdict = [(cmin, cmax) for _ in range(size)]

                elif costfn == 6:
                    cmin = np.random.uniform(1e-5, 1e-4, size=(size,))
                    cmax = np.random.uniform(1, 1, size=(size,))
                    costdict = [(cm, cx) for [cm, cx] in zip(cmin, cmax)]
                elif costfn == 7:
                    cmin = np.random.uniform(0, 0, size=(size,))
                    cmax = np.random.uniform(1, 1.5, size=(size,))
                    costdict = [(cm, cx) for [cm, cx] in zip(cmin, cmax)]
                elif costfn == 8:
                    cmin = np.random.uniform(0, 0, size=(size,))
                    cmax = np.random.uniform(0.5, 1, size=(size,))
                    costdict = [(cm, cx) for [cm, cx] in zip(cmin, cmax)]
                elif costfn == 9:
                    cmin = np.random.uniform(0, 0, size=(size,))
                    cmax = np.random.uniform(0.5, 1.5, size=(size,))
                    costdict = [(cm, cx) for [cm, cx] in zip(cmin, cmax)]
                elif costfn == 10:
                    cmin = np.random.uniform(1e-5, 1e-4, size=(size,))
                    cmax = np.random.uniform(1, 1.5, size=(size,))
                    costdict = [(cm, cx) for [cm, cx] in zip(cmin, cmax)]
                elif costfn == 11:
                    cmin = np.random.uniform(1e-5, 1e-4, size=(size,))
                    cmax = np.random.uniform(0.5, 1, size=(size,))
                    costdict = [(cm, cx) for [cm, cx] in zip(cmin, cmax)]
                elif costfn == 12:
                    cmin = np.random.uniform(1e-5, 1e-4, size=(size,))
                    cmax = np.random.uniform(0.5, 1.5, size=(size,))
                    costdict = [(cm, cx) for [cm, cx] in zip(cmin, cmax)]


                run(config, A, source, destination, costdict)



