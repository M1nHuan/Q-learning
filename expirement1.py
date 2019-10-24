import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Latex
from IPython.display import display
WORLD_HEIGHT = 4
WORLD_WIDTH = 12
GAMMA = 1
EPSILON = 0.1
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
START = [3, 0]
GOAL = [3, 11]

def print_optimal_policy(Q):
    ACTION_LATEX_LISTS = [r'\uparrow', r'\downarrow', r'\leftarrow', r'\rightarrow']
    policy_list = []
    for i in range(0, WORLD_HEIGHT):
        one_column_list = []
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                one_column_list.append('G')
                continue
            if i == 3 and 1 <= j <= 10:
                one_column_list.append('\square')
                continue
            max_a = np.argmax(Q[i, j])
            one_column_list.append(ACTION_LATEX_LISTS[max_a])
        one_column_str = '&'.join(one_column_list)
        policy_list.append(one_column_str)
    policy_str = r'$$\begin{bmatrix}' + r'\\'.join(policy_list) + r'\end{bmatrix}$$'
    display(Latex(policy_str))
class ENV:
    def __init__(self):
        self.START = [3, 0]
        self.GOAL = [3, 11]
        self.WORLD_WIDTH = WORLD_WIDTH
        self.WORLD_HEIGHT = WORLD_HEIGHT

    def step(self, s, a):
        i, j = s
        reward = -1
        if a == UP:
            next_S = [max(i - 1, 0), j]
        elif a == LEFT:
            next_S = [i, max(j - 1, 0)]
        elif a == RIGHT:
            next_S = [i, min(j + 1, self.WORLD_WIDTH - 1)]
        elif a == DOWN:
            next_S = [min(i + 1, self.WORLD_HEIGHT - 1), j]
        else:
            raise ValueError
        if next_S[0] == 3 and 1 <= next_S[1] <= 10:
            reward = -100
            next_S = self.START
        terminal = False
        if next_S == self.GOAL:
            terminal = True
        return next_S, reward, terminal

def epsilon_greedy(state,Q):
    if np.random.binomial(1,EPSILON) == 1:
        return np.random.choice(ACTIONS)
    else:
        values = Q[state[0],state[1]]
        return np.random.choice([a for a, v in enumerate(values) if v == np.max(values)])

def sarsa(env,Q,alpha = 0.5):
    s = env.START

    g = 0
    while s != env.GOAL:
        a = epsilon_greedy(s, Q)
        new_s,r,term = env.step(s,a)
        new_a = epsilon_greedy(new_s,Q)
        g += GAMMA * r
        td_target = r + GAMMA*Q[new_s[0],new_s[1],new_a]        #Q[s[0],s[1],a]
        Q[s[0],s[1],a] += alpha*(td_target - Q[s[0],s[1],a])
        s = new_s
        a = new_a
    return g

def Q_learning(env,Q,alpha = 0.5):
    s = env.START
    g = 0
    while s != env.GOAL:
        a = epsilon_greedy(s, Q)
        new_s,r,term = env.step(s, a)
        g += GAMMA* r
        td_target = r + GAMMA*np.max(Q[new_s[0],new_s[1]])
        Q[s[0], s[1], a] += alpha * (td_target - Q[s[0], s[1], a])
        s = new_s
    return g

def plot_qlearning(env,episodes = 1000 , N =50 ):
    mean_returns = np.zeros(episodes)
    for i in range(N):
        Q = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        for j in range(episodes):
            mean_returns[j] += Q_learning(env, Q)
    mean_returns /= N
    plt.plot(mean_returns)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.ylim([-100, 0])
    plt.show()
    print_optimal_policy(Q)


def main():
    episodes = 500
    N = 50
    env = ENV()
    """
    mean_returns = np.zeros(episodes)
    for i in range(N):
        Q = np.zeros((WORLD_HEIGHT,WORLD_WIDTH,4))
        for j in range(episodes):
            mean_returns[j] += sarsa(env,Q)
    mean_returns /= N
    plt.plot(mean_returns)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.ylim([-100, 0])
    plt.show()
    """
    plot_qlearning(env)

if __name__ == '__main__':
    main()