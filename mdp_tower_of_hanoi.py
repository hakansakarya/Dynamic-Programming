import random
import numpy as np
from copy import deepcopy
import time

class Tower_of_Hanoi(object):
	def __init__(self,state):
		#initialize Tower of Hanoi object
		self.num_of_disks = sum(1 for pin in state for disk in pin)
		self.pins = state


	def __eq__(self, other):
		#Method for object comparison
		for i, pin in enumerate(self.pins):
			if pin != other.pins[i]:
				return False
		return True
        
	def __str__(self):
		#to string method for printing
		pin1 = "+".join(str(x) for x in self.pins[0])
		pin2 = "+".join(str(x) for x in self.pins[1])
		pin3 = "+".join(str(x) for x in self.pins[2])
		return "Pin 1: {0} Pin 2: {1} Pin 3: {2}".format(pin1, pin2, pin3)

	def move_disk(self, action):
		#move disk from pin to pin given an action
		source_pin_idx, target_pin_idx = action
		pins = deepcopy(self.pins)
		disk = pins[source_pin_idx].pop()
		pins[target_pin_idx].append(disk)
		return Tower_of_Hanoi(pins)

	
	def is_goal_state(self):
		#returns whether a state is a goal state or not
		if len(self.pins[2]) == 2:
			last = self.pins[2][1]
			if last == 'B':
				return True
		return False

	def bigger_disk_on_top(self):
		#returns true if the bigger disk A is on top given the current state of the environment
		for pin in self.pins:
			if not pin == []:
				if pin[0] == 'B' and len(pin) == 2:
					return True
		return False

class MDP(object):
	def __init__(self):
		#initialize mdp environment
		self.gamma = 0.9
		self.states = [
						Tower_of_Hanoi([['A', 'B'], [], []]),
						Tower_of_Hanoi([['A'], ['B'], []]),
						Tower_of_Hanoi([['A'], [], ['B']]),
						Tower_of_Hanoi([[], ['A'], ['B']]),
						Tower_of_Hanoi([[], [], ['A', 'B']]),
						Tower_of_Hanoi([[], ['A', 'B'], []]),
						Tower_of_Hanoi([['B'], ['A'], []]),
						Tower_of_Hanoi([['B'], [], ['A']]),
						Tower_of_Hanoi([['B', 'A'], [], []]),
						Tower_of_Hanoi([[], ['B', 'A'], []]),
						Tower_of_Hanoi([[], [], ['B', 'A']]),
						Tower_of_Hanoi([[], ['B'], ['A']])
						]
		self.actions = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
		self.utilities = np.zeros(len(self.states))
		self.policies = [None for state in range(len(self.states))]

	def initialize_policies(self):
		#initialize policies randomly
		for index, state in enumerate(self.states):
			actions = self.get_actions(state)
			if len(actions) == 0: 
				continue
			self.policies[index] = random.choice(actions)

	def get_states(self):
		#return all states
		return self.states

	def get_actions(self, state):
		#return all available actions given a state
		if state.is_goal_state(): return []        
		actions = []
		for index, pin in enumerate(state.pins):
			for action in filter(lambda x: x[0] == index and len(pin) > 0, self.actions):
				actions.append(action)
		return actions

	def get_reward(self, state, action):
		#return the reward for taking an action in a given state
		if state.move_disk(action).is_goal_state():
			return 100
		elif state.move_disk(action).bigger_disk_on_top():
			return -10
		else:
			return -1

	def get_utility(self, state):
		#return the current utility of a state
		for index, s in enumerate(self.states):
			if s == state:
				return self.utilities[index]

	def update_utility(self, state, value):
		#update the utility value of a state
		for index, s in enumerate(self.states):
			if s == state:
				self.utilities[index] = value

	def update_policy(self, state, action):
		#update the policy for a state
		for index, s in enumerate(self.states):
			if s == state:
				self.policies[index] = action

	def get_policy_action(self, state):
		#get the action attached to the policy of the given state
		for index, s in enumerate(self.states):
			if s == state:
				return self.policies[index]

	def get_policies(self):
		#get all policies as a list
		return self.policies

	def get_transition_probabilities(self, state, action):
		#get the transition probablilites in the 
		#form of [(probability1, state1, action1), (probability2, state2, action2)]

		source_pin, target_pin = action
		wrong_pin = None
		
		if source_pin == 0:
			if target_pin == 1:
				wrong_pin = 2
			else:
				wrong_pin = 1

		elif source_pin == 1:
			if target_pin == 0:
				wrong_pin = 2
			else:
				wrong_pin = 0

		elif source_pin == 2:
			if target_pin == 1:
				wrong_pin = 0
			else:
				wrong_pin = 1

		return [(0.9, state.move_disk(action), action), (0.1, state.move_disk((source_pin, wrong_pin)), (source_pin, wrong_pin))]

	def value_iteration(self, epsilon):
		#value iteration algorithm
		while True:
			delta = 0
			for state in self.states:

				if self.get_actions == []: 
					continue

				state_utilities = []
				state_actions = []
                
				max_value = 0
				best_action = None

				for action in self.get_actions(state):
					transition_probs = self.get_transition_probabilities(state, action)
					reward = sum(p * self.get_reward(state, action) for (p, transition_state, action) in transition_probs)
					value = (reward) + (self.gamma * sum(p * self.get_utility(transition_state) for (p, transition_state, action) in transition_probs))
					if  value > max_value:
						max_value = value
						best_action = action
                
				delta = max(delta, abs(max_value - self.get_utility(state)))

				self.update_utility(state, max_value)
				self.update_policy(state, best_action)

			if delta <= epsilon:
				return self.get_policies()


	def policy_evaluation(self, epsilon = 10e-20):
		#policy evaluation
		while True:
			delta = 0
			for state in self.get_states():

				if self.get_actions(state) == []: 
					continue

				utility = self.get_utility(state)
				policy_action = self.get_policy_action(state)
				transition_probs = self.get_transition_probabilities(state, policy_action)
				updated_utility = sum(p * (self.get_reward(state, action) + self.gamma  * self.get_utility(transition_state)) for (p, transition_state, action) in transition_probs)
				self.update_utility(state, updated_utility)


			delta = max(delta, abs(updated_utility - utility))
			
			if delta < epsilon:
				break

	def policy_iteration(self):
		#initialize policies randomly
		self.initialize_policies()

		while True:
			#evaluate state utilities given current policy
			self.policy_evaluation()

			#policy imporvement
			unchanged = True
			for state in self.get_states():
				if self.get_actions(state) == []:
					continue
				state_utilities = []
				state_actions = []
				for action in self.get_actions(state):
					transition_probs = self.get_transition_probabilities(state, action)
					reward = self.get_reward(state, action)
					utility = reward + self.gamma * sum(p * self.get_utility(transition_state) for (p,transition_state,a) in transition_probs)
					state_utilities.append(utility)
					state_actions.append(action)
				max_index = state_utilities.index(max(state_utilities))
				maximizing_action = state_actions[max_index]
				if self.get_policy_action(state) != maximizing_action:
					self.update_policy(state, maximizing_action)
					unchanged = False
		
			if unchanged:
				return self.get_policies()


if __name__ == "__main__":
	
	mdp_hanoi1 = MDP()
	
	policy_iteration_start = time.time()
	optimal_policies = mdp_hanoi1.policy_iteration()
	policy_iteration_end = time.time()
	print('Policy iteration converged in: ', policy_iteration_end-policy_iteration_start, ' seconds.')

	for state in mdp_hanoi1.get_states():
		print('\n***Policy Iteration***')
		print('Actions are in the form of (Source Pin index, Target Pin index) with the index starting from 0.')
		print('State: ', state)
		print('Optimal Utility: ', mdp_hanoi1.get_utility(state))
		print('Optimal Policy: ', mdp_hanoi1.get_policy_action(state))
		print('Goal State: ', state.is_goal_state())
		print('Bigger disk is on top: ', state.bigger_disk_on_top())
		print('************************\n')
	
	print('\n\n\n')
	
	mdp_hanoi2 = MDP()		
	value_iteration_start = time.time()
	optimal_policies2 = mdp_hanoi2.value_iteration(10e-20)
	value_iteration_end = time.time()
	print('Value iteration converged in: ', value_iteration_end-value_iteration_start, ' seconds.')
	
	for state in mdp_hanoi2.get_states():
		print('\n***')
		print('Actions are in the form of (Source Pin index, Target Pin index) with the index starting from 0.')
		print('State: ', state)
		print('Available actions: ', mdp_hanoi2.get_actions(state))
		print('Optimal Policy: ', mdp_hanoi2.get_policy_action(state))
		print('Optimal Utility: ', mdp_hanoi2.get_utility(state))
		print('Goal State: ', state.is_goal_state())
		print('Bigger disk is on top: ', state.bigger_disk_on_top())
		print('***\n')
	
