
import gym
import numpy as np
from DataUtils import DataUtils

class ValueIterationExperiment(object):

    def __init__(self):

        self.discount_factor = 0.9
        self.convergence_threshold = 0.0001

    def run_problem_one_experiment(self):

        self.taxi_v2_env = gym.make('Taxi-v2')
        self.taxi_v2_env._max_episode_seconds = 999999999

        policy, V, iterations, theta, avg_deltas = self.value_iteration(env=self.taxi_v2_env.env, discount_factor=self.discount_factor, theta=self.convergence_threshold)

        score = self.evaluate_policy(self.taxi_v2_env, policy, self.discount_factor)

        print("Value Iteration converged on taxi problem -> \n\titerations to converge: " + str(iterations) + "\n\tconvergence threshold: " + str(theta) + "\n\tdiscount factor: " + str(self.discount_factor) + "\n\t100 game avg score: " + str(score) + "\n")

        DataUtils.write_convergence_diffs(DataUtils.get_results_directory_name() + "/taxi-value-iter-gamma-" + str(self.discount_factor) + ".csv", avg_deltas)

    def run_problem_two_experiment(self):

        self.frozen_lake_jumbo_env = gym.make('FrozenLakeJumbo-v0')

        policy, V, iterations, theta, avg_deltas = self.value_iteration(env=self.frozen_lake_jumbo_env, discount_factor=self.discount_factor, theta=self.convergence_threshold)

        score = self.evaluate_policy(self.frozen_lake_jumbo_env, policy, self.discount_factor)

        print("Value Iteration converged on frozen lake problem -> \n\titerations to converge: " + str(iterations) + "\n\tconergence threshold: " + str(theta) + "\n\tdiscount factor: " + str(self.discount_factor) + "\n\t100 game avg score: " + str(score) + "\n")

        DataUtils.write_convergence_diffs(DataUtils.get_results_directory_name() + "/lake-value-iter-gamma-" + str(self.discount_factor) + ".csv", avg_deltas)

    # Taken from https://www.kaggle.com/angps95/intro-to-reinforcement-learning-with-openai-gym
    def value_iteration(self, env, theta=0.0001, discount_factor=1.0):

        def one_step_lookahead(state, V):

            A = np.zeros(env.nA)
            for act in range(env.nA):
                for prob, next_state, reward, done in env.P[state][act]:
                    A[act] += prob * (reward + discount_factor * V[next_state])

            return A

        V = np.zeros(env.nS)
        iterations = 0
        avg_deltas = []
        while True:
            iterations += 1
            delta = 0  # checker for improvements across states
            for state in range(env.nS):
                act_values = one_step_lookahead(state, V)  # lookahead one step
                best_act_value = np.max(act_values)  # get best action value
                delta = max(delta, np.abs(best_act_value - V[state]))  # find max delta across all states
                V[state] = best_act_value  # update value to best action value
            if iterations > 1:
                avg_deltas.append(delta)
            if delta < theta:  # if max improvement less than threshold
                break

        policy = np.zeros([env.nS, env.nA])
        for state in range(env.nS):  # for all states, create deterministic policy
            act_val = one_step_lookahead(state, V)
            best_action = np.argmax(act_val)
            policy[state][best_action] = 1

        return policy, V, iterations, theta, avg_deltas

    # Taken from https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-9
    def run_episode(self, env, policy, gamma=1.0, render=False):

        obs = env.reset()
        total_reward = 0
        step_idx = 0
        while True:
            if render:
                env.render()
            obs, reward, done, _ = env.step(int(np.argmax(policy[obs])))
            total_reward += (gamma ** step_idx * reward)
            step_idx += 1
            if done or step_idx >= 50000:
                break

        return total_reward

    # Taken from https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-9
    def evaluate_policy(self, env, policy, gamma=1.0, n=100):

        scores = [
            self.run_episode(env, policy, gamma=gamma, render=False)
            for _ in range(n)]

        return np.mean(scores)