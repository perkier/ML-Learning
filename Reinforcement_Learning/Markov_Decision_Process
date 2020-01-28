import numpy as np
import random


class Environment:

    nan = np.nan  # Impossible actions

    def __init__(self):
        self.steps_left = 100

    def get_observation(self):
        return [0.0, 0.0, 0.0]

    def get_actions(self):
        return [[0, 1, 2], [0,2], [1]]

    def get_rewards(self):

        self.R = np.array([
            [[10, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[10, 0.0, 0.0], [self.nan, self.nan, self.nan], [0.0, 0.0, -50.0]],
            [[self.nan, self.nan, self.nan], [40.0, 0.0, 0.0], [self.nan, self.nan, self.nan]],
        ])
        return self.R

    def get_probabilities(self):

        self.T = np.array([
            [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
            [[0.0, 1.0, 0.0], [self.nan, self.nan, self.nan], [self.nan, self.nan, self.nan]],
            [[self.nan, self.nan, self.nan], [0.8, 0.1, 0.1], [self.nan, self.nan, self.nan]],
        ])
        return self.T

    def is_done(self):
        return self.steps_left == 0

    def action(self, action):

        if self.is_done():
            raise Exception("Game is over")

        self.steps_left -= 1

        return random.random()



def Q_value_iteration(discount_rate):

    Q = np.full( (3,3), -np.inf)  # -infinity for impossible actions

    env = Environment()

    possible_actions = env.get_actions()

    for state, actions in enumerate(possible_actions):
        Q[state, actions] = 0.0

    learning_rate = 0.01

    n_iterations = 100

    T = env.get_probabilities()
    R = env.get_rewards()

    for iterations in range(n_iterations):

        Q_prev = Q.copy()

        for s in range(3):
            for a in possible_actions[s]:
                for sp in range(3):

                    Q[s,a] = np.sum([T[s, a, sp] * (R[s, a, sp] + discount_rate * np.max(Q_prev[sp]))])


    return Q


def main():

    Q = Q_value_iteration(discount_rate = 0.95)

    print('\n' * 3)
    print(Q)
    print('\n' * 3)
    print(np.argmax(Q, axis=1))
    print('\n' * 3)

    Q = Q_value_iteration(discount_rate=0.99)

    print('\n' * 3)
    print(Q)
    print('\n' * 3)
    print(np.argmax(Q, axis=1))
    print('\n' * 3)




if __name__ == '__main__':
    main()
