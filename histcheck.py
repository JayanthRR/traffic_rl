from rl import *
from graphs import *
np.random.RandomState(seed=12356)


def hist_of_states(transition_matrix, x_init, num_steps):

    aggr_states = np.array(x_init)
    xt = x_init
    for _ in range(num_steps):
        xt = state_transition(transition_matrix, xt, None)
        aggr_states = np.hstack((aggr_states, xt))
    return aggr_states


size = 50
x_init = np.random.rand(size)
x_init = x_init #/ x_init.sum()
transition_matrix = generate_column_stochastic_matrix(size)
destination = np.random.randint(size)
source = np.random.randint(size)
env = TrafficEnv(transition_matrix, x_init, source, destination, quantize=False, fine=0.01)
num_steps = 1000

aggr_states = hist_of_states(env.transition_matrix, x_init, num_steps)
plt.hist(aggr_states, bins='auto')
plt.show()
hist, bin_edges = np.histogram(aggr_states)

