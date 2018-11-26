
import gym
import numpy as np

class PolicyIterationExperiment(object):

    def __init__(self):

        self.discount_factor = 0.9

    def run_problem_one_experiment(self):

        self.taxi_v2_env = gym.make('Taxi-v2')
        self.taxi_v2_env._max_episode_seconds = 999999999

        policy, iterations = self.policy_iteration(env=self.taxi_v2_env.env, discount_factor=self.discount_factor)

        score = self.evaluate_policy(self.taxi_v2_env, policy, self.discount_factor)

        print("Policy Iteration converged on taxi problem -> \n\titerations to converge: " + str(iterations) + "\n\tdiscount factor: " + str(self.discount_factor) + "\n\t100 game avg score: " + str(score) + "\n")

    def run_problem_two_experiment(self):

        self.frozen_lake_jumbo_env = gym.make('FrozenLakeJumbo-v0')

        policy, iterations = self.policy_iteration(env=self.frozen_lake_jumbo_env, discount_factor=self.discount_factor)

        score = self.evaluate_policy(self.frozen_lake_jumbo_env, policy, self.discount_factor)

        print("Policy Iteration converged on frozen lake problem -> \n\titerations to converge: " + str(iterations) + "\n\tdiscount factor: " + str(self.discount_factor) + "\n\t100 game avg score: " + str(score) + "\n")

    # Taken from https://www.kaggle.com/angps95/intro-to-reinforcement-learning-with-openai-gym
    def policy_iteration(self, env, discount_factor=1.0):

        def one_step_lookahead(state, V):

            A = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[state][a]:
                    A[a] += prob * (reward + discount_factor * V[next_state])

            return A

        policy = np.ones([env.nS, env.nA]) / env.nA
        iterations = 0
        while True:
            iterations += 1
            curr_pol_val = self.policy_eval(policy, env, discount_factor)  # eval current policy
            policy_stable = True  # Check if policy did improve (Set it as True first)
            for state in range(env.nS):  # for each states
                chosen_act = np.argmax(policy[state])  # best action (Highest prob) under current policy
                act_values = one_step_lookahead(state, curr_pol_val)  # use one step lookahead to find action values
                best_act = np.argmax(act_values)  # find best action
                if chosen_act != best_act:
                    policy_stable = False  # Greedily find best action
                policy[state] = np.eye(env.nA)[best_act]  # update
            if policy_stable:
                break

        return policy, iterations

    # Taken from https://www.kaggle.com/angps95/intro-to-reinforcement-learning-with-openai-gym
    def policy_eval(self, policy, env, discount_factor=1.0, theta=0.0001):

        V = np.zeros(env.nS)
        while True:
            delta = 0  # delta = change in value of state from one iteration to next
            for state in range(env.nS):  # for all states
                val = 0  # initiate value as 0
                for action, act_prob in enumerate(policy[state]):  # for all actions/action probabilities
                    for prob, next_state, reward, done in env.P[state][
                        action]:  # transition probabilities,state,rewards of each action
                        val += act_prob * prob * (reward + discount_factor * V[next_state])  # eqn to calculate
                delta = max(delta, np.abs(val - V[state]))
                V[state] = val
            if delta < theta:  # break if the change in value is less than the threshold (theta)
                break

        return np.array(V)

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