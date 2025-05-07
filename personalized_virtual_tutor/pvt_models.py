# pvt_models.py

import numpy as np
import random

class QLearningTutor:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.q_table = {}
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))
        old = self.q_table[state][action]
        self.q_table[state][action] = old + self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state]) - old
        )

class Bandit:
    def __init__(self, actions, c=1.0):
        self.actions = actions
        self.c = c
        self.counts = [0] * len(actions)
        self.values = [0.0] * len(actions)

    def choose_action(self):
        total = sum(self.counts)
        # Ensure each arm is tried once
        for i in range(len(self.actions)):
            if self.counts[i] == 0:
                return i
        ucb = [
            self.values[i] + self.c * np.sqrt(np.log(total) / self.counts[i])
            for i in range(len(self.actions))
        ]
        return int(np.argmax(ucb))

    def update(self, action_index, reward):
        self.counts[action_index] += 1
        n = self.counts[action_index]
        value = self.values[action_index]
        # incremental update
        self.values[action_index] = value + (reward - value) / n
