
import numpy as np
import random
import gym
from DataUtils import DataUtils

class QLearningExperiment(object):

    def __init__(self):

        self.discount_factor = 0.9
        self.convergence_threshold = 0.0001

    def run_problem_one_experiment(self):

        self.taxi_v2_env = gym.make('Taxi-v2')
        self.taxi_v2_env._max_episode_seconds = 999999999

        q_table, iterations, avg_deltas = self.q_learning(self.taxi_v2_env)

        score = self.evaluate_policy(env=self.taxi_v2_env, q_table=q_table, gamma=self.discount_factor)

        print("Q-Learning converged on taxi problem -> \n\titerations to converge: " + str(iterations) + "\n\tconvergence threshold: " + str(self.convergence_threshold) + "\n\tdiscount factor: " + str(self.discount_factor) + "\n\t100 game avg score: " + str(score) + "\n")

        DataUtils.write_convergence_diffs(DataUtils.get_results_directory_name() + "/taxi-q-learning-gamma-" + str(self.discount_factor) + ".csv", avg_deltas)

    def run_problem_two_experiment(self):

        self.frozen_lake_jumbo_env = gym.make('FrozenLakeJumbo-v0')

        q_table, iterations, avg_deltas = self.q_learning(self.frozen_lake_jumbo_env, 2000000)

        score = self.evaluate_policy(env=self.frozen_lake_jumbo_env, q_table=q_table, gamma=self.discount_factor)

        print("Q-Learning converged on frozen lake problem -> \n\titerations to converge: " + str(iterations) + "\n\tconvergence threshold: " + str(self.convergence_threshold) + "\n\tdiscount factor: " + str(self.discount_factor) + "\n\t100 game avg score: " + str(score) + "\n")

        DataUtils.write_convergence_diffs(DataUtils.get_results_directory_name() + "/lake-q-learning-gamma-" + str(self.discount_factor) + ".csv", avg_deltas)

    def q_learning(self, env, min_iter=100000):

        in_dimen = env.observation_space.n
        out_dimen = env.action_space.n
        q_table = np.zeros(shape=(in_dimen, out_dimen))
        total_reward = 0
        epsilon = 1.0
        alpha = 0.2
        epsilon_decay_rate = 0.999995
        gamma = self.discount_factor

        iterations = 0
        avg_deltas = []

        converged = False

        while not converged:

            current_state = env.reset()
            current_deltas = []

            while True:

                iterations += 1

                prevState = current_state

                epsilon *= epsilon_decay_rate

                if epsilon < 0.1:
                    epsilon = 0.1

                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[current_state])

                current_state, reward, done, info = env.step(action)

                total_reward += reward

                delta = q_table[prevState][action]

                if reward > 0:
                    q_table[prevState][action] += alpha * (reward + (gamma * 0) - q_table[prevState][action])
                else:
                    q_table[prevState][action] += alpha * (reward + (gamma * max(q_table[current_state])) - q_table[prevState][action])

                delta = abs(delta - q_table[prevState][action])

                if iterations > 1:
                    avg_deltas.append(delta)
                    current_deltas.append(delta)

                if done:
                    break

            if iterations >= min_iter:
                found_big_delta = False
                for i in range(len(current_deltas)):
                    if current_deltas[i] > self.convergence_threshold:
                        found_big_delta = True
                        break
                if not found_big_delta:
                    converged = True
                    break

        return q_table, iterations, avg_deltas

    # Adapted from https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-9
    def run_episode(self, env, q_table, gamma=1.0, render=False):

        obs = env.reset()
        total_reward = 0
        step_idx = 0
        while True:
            if render:
                env.render()
            obs, reward, done, _ = env.step(int(np.argmax(q_table[obs])))
            total_reward += (gamma ** step_idx * reward)
            step_idx += 1
            if done or step_idx >= 50000:
                break

        return total_reward

    # Adapted from https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-9
    def evaluate_policy(self, env, q_table, gamma=1.0, n=100):

        scores = [
            self.run_episode(env, q_table, gamma=gamma, render=False)
            for _ in range(n)]

        return np.mean(scores)