README for submission of mpalmer47 / Matt Palmer / GTID: 903336804
for Project 4 - Markov Decision Processes for CS 7641 Fall 2018.

Source Code Repository:

    - https://github.com/Mattjp213/mpalmer47-project4.git

Dependencies:

	- Python 3.6
	- numpy 1.15.4
	- gym 0.10.9

Source Code Files:

	- Application.py (contains main method of application)
	- DataUtils.py (contains static utility methods for writing results to file system)
	- ValueIterationExperiment.py (contains methods for running value iteration on both environments)
	- PolicyIterationExperiment.py (contains methods for running policy iteration on both environments)
	- QLearningExperiment.py (contains methods for running q-learning on both environments)
	- frozen_lake_jumbo.py (contains code for custom open ai gym environment - 32 x 32 frozen lake)

Running the Application:

	- Running the Application.py file will run any uncommented experiments included in the project automatically. There
	  is 1 call to run each of the individual experiments in the main method of this file and each one is aptly named
	  and also described in a comment directly above each respective method call. The problems are referred to as 'one'
	  and 'two' in the comments in Application.py. Problem one refers to the taxi-v2 problem and problem two refers to
	  the custom frozen lake problem that I created for this project. Currently, only the method calls to run value
	  iteration on each of the two problems is uncommented. You can uncomment other calls to run the other experiments.
	  In my paper, I show results for various discount factors that were used. These values can be configured in the
	  constructor for each of the experiment classes. Other than that, everything needed to run the experiments is in
	  the main method of the Application.py file and does not need to be modified to reproduce the results shown in my
	  paper. When you run Application.py it will automatically create a folder called 'Results' in the root directory of
	  the project if it does not already exist. The value iteration and q-learning experiments will generate CSV files
	  in this folder containing the maximum value function or q-table differences between iterations to show the
	  progress of convergence over time. I used these CSV files to produce my graphs in Excel. Every experiment will
	  output some information to the Console describing the results of the experiment. Screenshots of this Console
	  output can also be found in my paper. When you run Application.py the custom frozen lake environment that I
	  created is automatically registered with open ai gym. This should work without any modification or setup other
	  than making sure that open ai gym is installed in you virtual environment.

After Word:

    - Thanks for a great semester. It was very challenging, but I feel like I learned a lot.