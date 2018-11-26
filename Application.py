
# Matt Palmer, mpalmer47@gatech.edu, GTID: 903336804
# CS 7641 Fall 2018 Project 4

from ValueIterationExperiment import ValueIterationExperiment
from PolicyIterationExperiment import PolicyIterationExperiment
from QLearningExperiment import QLearningExperiment
from DataUtils import DataUtils
import gym
import os

def run_value_iteration_on_problem_one():
    value_iteration_exp = ValueIterationExperiment()
    value_iteration_exp.run_problem_one_experiment()

def run_value_iteration_on_problem_two():
    value_iteration_exp = ValueIterationExperiment()
    value_iteration_exp.run_problem_two_experiment()

def run_policy_iteration_on_problem_one():
    policy_iteration_exp = PolicyIterationExperiment()
    policy_iteration_exp.run_problem_one_experiment()

def run_policy_iteration_on_problem_two():
    policy_iteration_exp = PolicyIterationExperiment()
    policy_iteration_exp.run_problem_two_experiment()

def run_q_learning_on_problem_one():
    q_learning_exp = QLearningExperiment()
    q_learning_exp.run_problem_one_experiment()

def run_q_learning_on_problem_two():
    q_learning_exp = QLearningExperiment()
    q_learning_exp.run_problem_two_experiment()

if __name__ == "__main__":

    print("Application running...\n")

    # Problem one is the small state space problem and problem two is the large state space problem.

    # Make sure that the 'Results' directory is present to save output files in.
    if not os.path.isdir(DataUtils.get_results_directory_name()):
        os.mkdir(DataUtils.get_results_directory_name())

    # Run value iteration on the first problem.
    run_value_iteration_on_problem_one()

    # Run policy iteration on the first problem.
    #run_policy_iteration_on_problem_one()

    # Run q learning on the first problem.
    #run_q_learning_on_problem_one()

    # Register the custom jumbo lake environment with open ai gym.
    gym.envs.registration.register(
        id='FrozenLakeJumbo-v0',
        entry_point='frozen_lake_jumbo:FrozenLakeJumboEnv',
    )

    # Run value iteration on the second problem.
    run_value_iteration_on_problem_two()

    # Run policy iteration on the second problem.
    #run_policy_iteration_on_problem_two()

    # Run q learning on the second problem.
    #run_q_learning_on_problem_two()