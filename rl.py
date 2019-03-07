import numpy as np
from random import choice
# from matplotlib import pyplot as plt
import pickle
from graphs import *
# np.random.RandomState(seed=10)
import copy
import itertools

marker = itertools.cycle(('X', '+', 'o', '.', '*', '-', '1', '2', '3', '4', '5'))


def longest_destination(transition_matrix, source, min_len=None):
    max_len = transition_matrix.shape[0]
    path_lens = [None]*max_len
    queue1 = []
    queue1.insert(0, source)
    path_lens[source] = 0
    path_len = 0
    queue2 = []
    while queue1:
        path_len += 1
        while queue1:
            current_edge = queue1.pop()
            next_edges = get_successor_edges(transition_matrix, current_edge)
            for edge in next_edges:
                if path_lens[edge] is None:
                    path_lens[edge] = path_len
                    queue2.insert(0, edge)
                if min_len:
                    if path_len == min_len:
                        return edge, min_len

        queue1 = queue2
        queue2 = []

    ret = 0
    max_so_far = 0
    for ind in range(max_len):
        if path_lens[ind]:
            if path_lens[ind] > max_so_far:
                ret = ind
                max_so_far = path_lens[ind]

    return ret, path_lens[ret]


def generate_column_stochastic_matrix(n):
    # Doesn't generate a sparse matrix. The degree of each link can be as high as n-1.
    # Todo: Think of how to make a sparse matrix (to mimic road network geometry)

    # A = np.random.rand(n,n)
    # temp = np.random.choice([0,1], (n,n), p=[0.9,0.1])
    # A = np.multiply(A, temp)
    #
    # A = A/A.sum(axis=0)

    A = np.zeros((n, n))
    for i in range(n):
        while True:
            temp = np.random.rand(n)
            mask = np.random.choice([0,1], (n), p=[0.99,0.01])
            temp = temp * mask
            if temp.sum() > 0:
                A[:, i] = temp/temp.sum()
                break
    return A


def state_transition(A, xt, wt, mean=0, sigma=0.1):
    c = 1
    if wt is None:
        wt = 1 * gaussian(len(xt), mean, sigma)

    x_t1 = np.matmul(A, xt) + wt

    x_t1 = np.minimum(np.maximum(x_t1, np.zeros(len(xt))), c * np.ones(len(xt)))

    return x_t1


def gaussian(k, mean=0, sigma=0.01):
    return np.random.normal(loc=mean, scale=sigma, size=k)


def epsilon_greedy(epsilon=0.2):
    # epsilon close to 0.
    if np.random.random() > epsilon:
        return True
    else:
        return False


def get_bin_endings(min=0.0, max=1.0, size=15, type="log"):
    bin_endings = []
    temp = max

    if type == "log":
        bin_endings.append(max)
        for ind in range(size - 1):
            temp = (temp - min)/2
            bin_endings.append(temp)
        return list(reversed(bin_endings))

    if type == "uniform":
        for ind in range(size):
            bin_endings.append(min + (1 + ind)*(max-min)/size)
        return bin_endings


class TrafficEnv:
    def __init__(self, transition_matrix, x_init, source, destination, quantize=True, qlen=10,
                 type="uniform"):
        self.transition_matrix = transition_matrix
        self.xt = x_init
        self.source = source
        self.destination = destination
        self.current_edge = source
        self.prev_edge = None
        self.size = len(x_init)
        self.actions = list(range(self.size))
        self.current_edge_one_hot = np.zeros((len(x_init)))
        self.current_edge_one_hot[self.source] = 1
        self.qlen = qlen
        self.quantized_xt = np.zeros(self.size*qlen)
        self.type_ = type
        self.quantize_bin_endings = get_bin_endings(min=0, max=1, size=qlen, type=type)
        self.quantize_state()
        self.quantize_flag = quantize
        if not self.quantize_flag:
            self.state = np.hstack((self.xt, self.current_edge_one_hot))
            self.q_function = QFunction(len(self.state), self.size)
        else:
            self.state = np.hstack((self.quantized_xt, self.current_edge_one_hot))/(self.size + 1)
            self.q_function = QFunction(len(self.state), self.size)

    def reset(self):
        self.current_edge = self.source
        self.prev_edge = None
        self.current_edge_one_hot = np.zeros((len(self.xt)))
        self.current_edge_one_hot[self.source] = 1
        self.random_init()
        if not self.quantize_flag:
            self.state = np.hstack((self.xt, self.current_edge_one_hot))
        else:
            self.state = np.hstack((self.quantized_xt, self.current_edge_one_hot))/(self.size + 1)

    def random_init(self):
        np.random.seed()
        self.xt = np.random.rand(self.size)
        self.xt = self.xt/self.xt.sum()
        self.quantize_state()

    def quantize_state(self):
        self.quantized_xt = np.zeros(self.size*self.qlen)
        for i in range(self.size):
            for ind in range(self.qlen):
                if self.xt[i] > self.quantize_bin_endings[ind]:
                    continue
                else:
                    break

            self.quantized_xt[i*self.qlen + ind] = 1/(self.size + 1)

    def get_successor_edges(self, current_edge=None):
        adj = []
        if not current_edge:
            current_edge = self.current_edge
        for i in range(self.transition_matrix.shape[0]):
            if self.transition_matrix[i, current_edge] > 0:
                if i == current_edge:
                    continue
                adj.append(i)
        return adj

    def get_predecessor_edges(self, current_edge=None):
        adj = []
        if not current_edge:
            current_edge = self.current_edge
        for i in range(self.transition_matrix.shape[0]):
            if self.transition_matrix[current_edge, i] > 0:
                if i == current_edge:
                    continue
                adj.append(i)
        return adj

    def get_reward(self, action):

        if action == self.destination:
            return 0
        else:
            return -cost_function(self.xt[self.current_edge])

    def state_transition(self, noise_t=None):
        c = 1

        if noise_t is None:
            noise_t = gaussian(self.size)
        temp = np.matmul(self.transition_matrix, self.xt) + c * noise_t
        self.xt = np.minimum(np.maximum(temp, np.zeros(self.size)), np.ones(self.size))
        self.quantize_state()

    def state_update(self):
        self.current_edge_one_hot[self.prev_edge] = 0
        self.current_edge_one_hot[self.current_edge] = 1/(self.size + 1)
        if self.quantize_flag:
            self.state = np.hstack((self.quantized_xt, self.current_edge_one_hot))
        else:
            self.state = np.hstack((self.xt, self.current_edge_one_hot))

    def get_next_state_reward(self, action):
        # self.state_transition()
        reward = self.get_reward(action)
        self.prev_edge = self.current_edge
        self.current_edge = action
        self.state_update()

        return self.state, reward


class TrafficAgent:
    def __init__(self, env, learning_rate, epsilon, exploration_decay):
        self.env = env
        self.learning_rate = learning_rate
        self.exploration_decay = exploration_decay
        self.q_function = QFunction(len(self.env.state), self.env.size)
        self.epsilon = epsilon

    def get_action(self):
        candidate_actions = self.env.get_successor_edges(current_edge=self.env.current_edge)
        if not candidate_actions:
            return None

        return self.q_function.argmax(self.env.state, candidate_actions)

    def softmax_policy(self):
        candidate_actions = self.env.get_successor_edges(current_edge=self.env.current_edge)
        if not candidate_actions:
            return None

        values = [self.q_function.evaluate(self.env.state, action) for action in candidate_actions]
        probs = [np.exp(value) for value in values]
        probs /= sum(probs)
        return candidate_actions, probs

    def get_softmax_action(self):
        candidate_actions, probs = self.softmax_policy()
        rand = np.random.rand()
        cum_sum = 0
        for ind, action in enumerate(candidate_actions):
            cum_sum += probs[ind]
            if cum_sum >= rand:
                return action

    def get_epsilon_greedy_action(self, max=False):
        candidate_actions = self.env.get_successor_edges(current_edge=self.env.current_edge)
        if not candidate_actions:
            return None

        if max:
            return self.q_function.argmax(self.env.state, candidate_actions)

        if epsilon_greedy(self.epsilon):
            return self.q_function.argmax(self.env.state, candidate_actions)
        else:
            return choice(candidate_actions)

    def gradient_update(self, action, target_diff, state):

        self.q_function.W[:, action] += self.learning_rate * target_diff * self.q_function.gradient(state, action)

        flag = np.isnan(np.sum(self.q_function.W))
        if flag:
            print("hmm")


class QFunction:
    def __init__(self, state_space_dim, action_space_dim, random_init=False):

        if random_init:
            self.W = np.random.normal(size=(state_space_dim, action_space_dim))
        else:
            self.W = np.zeros((state_space_dim, action_space_dim))

        # self.W_ind = np.zeros((state_space_dim, state_space_dim))

    def evaluate(self, state, action):
        # print(xt.shape, self.W[:, action].shape)
        temp = np.matmul(state, self.W[:, action])
        # if np.isinf(temp):
        #     print("how??")

        return temp

    def update_w(self, action, w_action):
        self.W[:, action] = w_action

    def argmax(self, state, actions, ret_max=False):
        best = -np.inf
        best_action = None
        for action in actions:
            temp = self.evaluate(state, action)
            # print(best, temp)
            if temp > best:
                best = temp
                best_action = action
        if best_action is None:
            print("check point")

        if ret_max:
            return best_action, best
        else:
            return best_action

    def gradient(self, state, action):
        return state


def sarsa(num_episodes, agent, gamma, expected=False, softmax=False):
    # don't use. Use train agent with qlearning False instead
    episode = 0

    MAX_W = agent.q_function.W.size
    MIN_W = -agent.q_function.W.size
    rewards = []
    while episode < num_episodes:
        spath = []
        print("e",episode)
        episode_rewards = []
        e_reward = 0
        # get initial state and action
        agent.env.reset()
        state = agent.env.state
        candidate_edges = agent.env.get_successor_edges()
        if not candidate_edges:
            episode += 1
            continue

        action = choice(candidate_edges)
        steps = 0
        while True:
            agent.env.state_transition()
            next_state, reward = agent.env.get_next_state_reward(action)
            spath.append(action)
            e_reward += reward
            episode_rewards.append(e_reward) #+= reward
            if action == agent.env.destination:
                target_diff = reward - agent.q_function.evaluate(state, action)
                if target_diff > MAX_W:
                    target_diff = MAX_W
                if target_diff < MIN_W:
                    target_diff = MIN_W

                agent.gradient_update(action, target_diff, state)
                # agent.q_function.W[len(env.quantized_xt) + agent.env.prev_edge, action] = agent.q_function.W.size
                print("found")
                break

            if softmax:
                next_action = agent.get_softmax_action()
            else:
                next_action = agent.get_epsilon_greedy_action()

            if next_action is None:
                # target_diff = -1e6
                # agent.gradient_update(action, target_diff)
                agent.q_function.W[len(state) - agent.env.size + agent.env.prev_edge, action] = -agent.q_function.W.size
                print("destination not found")
                break

            if expected:
                candidate_actions, probs = agent.softmax_policy()
                target_diff = reward - agent.q_function.evaluate(state, action)

                for ind in range(len(candidate_actions)):
                    target_diff += gamma * agent.q_function.evaluate(next_state, candidate_actions[ind]) * probs[ind]
            else:
                target_diff = reward + gamma*agent.q_function.evaluate(next_state, next_action) - \
                          agent.q_function.evaluate(state, action)

            if target_diff > MAX_W:
                target_diff = MAX_W
            if target_diff < MIN_W:
                target_diff = MIN_W

            # if np.isinf(target_diff):
            #     print("how??")
            agent.gradient_update(action, target_diff, state)
            state = next_state
            action = next_action
            steps += 1
        # plt.plot(episode_rewards)

        rewards.append(episode_rewards)
        episode += 1
        agent.epsilon = agent.epsilon * agent.exploration_decay
        print(len(spath))

    plt.plot([len(r) for r in rewards])
    # plt.show()
    plt.plot([r[-1] for r in rewards])
    # plt.plot([r[-1] for r in rewards[:100]]+[r[-1] for r in rewards[-100:]])
    # plt.show()
    plt.savefig("reweards and num steps")
    plt.close()
    return agent


def train_agent(agent, num_episodes, discount_factor,
                expected=False, softmax=False, qlearning=True):

    episode = 0
    MAX_W = agent.q_function.W.size
    MIN_W = -agent.q_function.W.size
    training_rewards = []
    total_rewards = []

    while episode < num_episodes:
        episode_path = []

        episode_rewards = []
        total_reward = 0
        agent.env.reset()
        state = agent.env.state
        candidate_edges = agent.env.get_successor_edges()

        if not candidate_edges:
            break

        action = choice(candidate_edges)
        steps = 0
        while True:
            agent.env.state_transition()
            next_state, step_reward = agent.env.get_next_state_reward(action)
            episode_path.append(action)
            episode_rewards.append(step_reward)
            total_reward += step_reward

            if action == agent.env.destination:
                target_diff = step_reward - agent.q_function.evaluate(state, action)
                if target_diff > MAX_W:
                    target_diff = MAX_W
                if target_diff < MIN_W:
                    target_diff = MIN_W

                agent.gradient_update(action, target_diff, state)
                # print("found")
                break

            if qlearning:
                next_action = agent.get_epsilon_greedy_action(max=True)
                if next_action is None:
                    agent.q_function.W[len(state) - agent.env.size +
                                       agent.env.prev_edge, action] = -agent.q_function.W.size
                    print("destination not found")
                    break

                target_diff = step_reward + discount_factor*agent.q_function.evaluate(next_state, next_action) - \
                              agent.q_function.evaluate(state, action)

            else:

                if softmax:
                    next_action = agent.get_softmax_action()
                else:
                    next_action = agent.get_epsilon_greedy_action()

                if next_action is None:
                    agent.q_function.W[len(state) - agent.env.size +
                                       agent.env.prev_edge, action] = -agent.q_function.W.size
                    print("destination not found")
                    break

                if expected:
                    candidate_actions, probs = agent.softmax_policy()
                    target_diff = step_reward - agent.q_function.evaluate(state, action)

                    for ind in range(len(candidate_actions)):
                        target_diff += discount_factor * agent.q_function.evaluate(next_state, candidate_actions[ind]) * probs[
                            ind]
                else:
                    target_diff = step_reward + discount_factor * agent.q_function.evaluate(next_state, next_action) - \
                                  agent.q_function.evaluate(state, action)

            if target_diff > MAX_W:
                target_diff = MAX_W
            if target_diff < MIN_W:
                target_diff = MIN_W

            agent.gradient_update(action, target_diff, state)

            state = next_state
            if qlearning:
                next_action = agent.get_epsilon_greedy_action(max=True)

            action = next_action
            steps += 1

        training_rewards.append(episode_rewards)
        total_rewards.append(total_reward)

        if episode % 100 == 0:
            print("episode: ", episode, len(episode_path))

        episode += 1
        agent.epsilon = agent.epsilon * agent.exploration_decay

    # plt.plot([len(r) for r in training_rewards], marker=next(marker), label="num steps")
    # plt.plot(total_rewards, marker=next(marker), label="total rewards")
    # plt.legend()
    # plt.savefig('plots/out.pdf', transparent=True, bbox_inches='tight', pad_inches=0)
    # plt.close()
    return agent, training_rewards, total_rewards


def evaluate_policies(env, W, policy="dijkstra", lookahead=0, const_flag=False):

    if policy == "dijkstra":
        control = dijkstra_policy
    else:
        control = greedy

    const_path = []
    aggr_reward = list()
    path_taken = list()
    path_taken.append(env.source)
    reward_incurred = 0
    aggr_reward.append(reward_incurred)
    count = 0
    while env.current_edge is not env.destination:

        # for i in range(lookahead):
        #     env.state_transition()
        xt = env.xt
        for look in range(lookahead):
            xt = state_transition(env.transition_matrix, xt, env.current_edge, env.destination)

        if const_flag:
            if not const_path:
                for look in range(lookahead):
                    xt = state_transition(env.transition_matrix, xt, env.current_edge, env.destination)

                const_path = const_dijkstra_policy(env.transition_matrix, xt, env.current_edge, env.destination)
                if const_path:
                    decision = const_path.pop()
                else:
                    print(policy, "no path")
                    reward_incurred = -inf
                    break

            else:
                decision = const_path.pop()
        else:
            decision = control(env.transition_matrix, xt, env.current_edge, env.destination)

        path_taken.append(decision)

        if decision is None:
            print(policy, "no path")
            reward_incurred = -inf
            break

        wt = W[count]

        env.state_transition(noise_t=wt)
        next_state, reward = env.get_next_state_reward(decision)

        reward_incurred += reward
        aggr_reward.append(reward_incurred)
        count += 1
        if count == len(env.xt):
            break

    return reward_incurred, aggr_reward, path_taken


def evaluate_rl_policy(traffic_agent, W):

    aggr_reward = list()
    path_taken = list()
    path_taken.append(traffic_agent.env.source)
    reward_incurred = 0
    aggr_reward.append(reward_incurred)
    count = 0

    while traffic_agent.env.current_edge is not traffic_agent.env.destination:
        # Todo: signal flag

        decision = traffic_agent.get_action()

        path_taken.append(decision)

        if decision is None:
            print("no path")
            reward_incurred = -np.inf
            break

        wt = W[count]

        traffic_agent.env.state_transition(noise_t=wt)
        next_state, reward = traffic_agent.env.get_next_state_reward(decision)

        reward_incurred += reward
        aggr_reward.append(reward_incurred)
        count += 1
        if count == 5*len(wt):
            print("exceeded time limit")
            break

    return reward_incurred, aggr_reward, path_taken


def main():

    marker_1 = itertools.cycle(('X', '+', 'o', '.', '*', '-', '1', '2', '3', '4', '5'))

    num_episodes = 20000
    epsilon = 0.5
    gamma = 1
    learning_rate = 0.1

    size = 100
    x_init = np.random.rand(size)
    x_init = x_init/x_init.sum()
    transition_matrix = generate_column_stochastic_matrix(size)
    source = np.random.randint(size)
    destination, min_len = longest_destination(transition_matrix, source, min_len=int(size/4))
    # destination = np.random.randint(size)
    print(source, destination, min_len)

    while min_len <= 2:
        transition_matrix = generate_column_stochastic_matrix(size)
        source = np.random.randint(size)
        destination, min_len = longest_destination(transition_matrix, source, min_len=int(size / 4))

    env = TrafficEnv(transition_matrix, x_init, source, destination, quantize=False, type="log")
    agent = TrafficAgent(env, learning_rate, epsilon, exploration_decay=0.99)

    agent = train_agent(agent, num_episodes, gamma, softmax=False, expected=False, qlearning=False)

    num_trials = 100
    avg_reward = 0
    avg_reward_rl = 0
    path_rl_total = []
    path_total = []
    aggr_reward_rl = []
    aggr_reward = []

    for ind in range(num_trials):
        W = []
        time_steps = 2*size
        for i in range(time_steps):
            W.append(gaussian(size, 0, 0.1))

        env.reset()
        env_copy = copy.deepcopy(env)
        agent.env = env
        reward_rl, _, path_rl = evaluate_rl_policy(agent, W)
        # reward_rl, _, path_rl = evaluate_policies(env, W, policy="dijkstra", lookahead=5)
        reward, _, path = evaluate_policies(env_copy, W, lookahead=0)
        avg_reward += reward
        avg_reward_rl += reward_rl
        path_rl_total.append(path_rl)
        path_total.append(path)
        aggr_reward.append(reward)
        aggr_reward_rl.append(reward_rl)

    avg_reward_rl /= num_trials
    avg_reward /= num_trials
    plt.close()
    plt.plot(aggr_reward, color="red", label="dijkstra", marker=next(marker_1))
    plt.plot(aggr_reward_rl, color="green", label="rl", marker=next(marker_1))
    plt.legend()
    plt.savefig('plots/comparision_rewards.pdf', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.plot([len(p) for p in path_total], color="red", label="dijkstra", marker=next(marker_1))
    plt.plot([len(p) for p in path_rl_total], color="green", label="rl", marker=next(marker_1))
    plt.legend()
    plt.savefig('plots/comparision step lengths.pdf', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()
    # plt.show()
    # pickle.dump(new_traffic_control, open("luckylog.p", "wb"))
    print("hurray")


if __name__ == "__main__":

    main()
