# EEG-TAMER
This is an implementation of TAMER which uses EEG signals from a 16-channel EMOTIV EPOC + as the positive and negative training input. 6 high perfomance metrics are available as training metrics. 
A basic clicker-TAMER implementation was used from a repo, documentation at the bottom of this read me (TAMER, .. ) credits to Benibienz (https://github.com/benibienz/TAMER). 

## Getting started:
-	Pull from github: https://github.com/SherbertLemon77/EEG-TAMER
-	Install EMOTIV Launcher and log in with credentials: https://www.emotiv.com/emotiv-launcher/
o	User:  --- confidential please ask Lisa, Rohan or Esi
o	Password: --- confidential please ask Lisa, Rohan or Esi
-	Have headset on and connected while coding (so authentication goes through and EEG stream is running)
-	To run EEG-TAMER system execute ./run.py file from command line
	 
## Goal of Code: 
Connect EEG to TAMER to train Mountain Car

## Code outline: 
Run Mountain Car Simulation, Run TAMER, authenticate to EEG and get a stream of high performance metrics into a queue, TAMER takes metrics off of this queue and uses them as feedback (multiprocessed)

### 3 main files of interest: 
run.py		(location: EEG TAMER>)
agent.py	(location: EEG TAMER>tamer)
interface.py	(location: EEG TAMER>tamer)

### main modules used: 
	gym
	asyncio
	multiprocessing
	queue
	threading
  	numpy
  	sklearn
  	pygame
	
# TAMER
TAMER (Training an Agent Manually via Evaluative Reinforcement) is a framework for human-in-the-loop Reinforcement Learning, proposed by [Knox + Stone](http://www.cs.utexas.edu/~sniekum/classes/RLFD-F16/papers/Knox09.pdf) in 2009. 

This is an implementation of a TAMER agent, converted from a standard Q-learning agent using the steps provided by Knox [here](http://www.cs.utexas.edu/users/bradknox/kcap09/Knox_and_Stone,_K-CAP_2009.html).



## How to run
You need python 3.7+ with numpy, sklearn, pygame and gym.

Use run.py. You can fiddle with the config in the script.

In training, watch the agent play and press 'W' to give a positive reward and 'A' to give a negative. The agent's current action is displayed.

![Screenshot of TAMER](screenshot.png)
