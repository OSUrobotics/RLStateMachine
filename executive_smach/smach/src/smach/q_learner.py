#!/usr/bin/env python
import time
import operator
import random
import rospy
from smach_msgs.msg import StateActionUpdates
import pylab
import numpy as np
from scipy import stats
from smach_msgs.srv import StateActionUpdatesService


class QLearner():
	def __init__(self):
		self.once = False
		self.q_learning = True
		self.additive_states = True
		self.current_iteration = 0

		self.q_table = dict(dict())
		self.q_table_additive_states = dict(dict())
		self.additive_states_prob = dict(dict())
		self.additive_states_prob_counts = dict(dict())

		self.learning_rate = 0.1
		self.discount_factor = 0.9
		self.states_to_update = list()
		rospy.Subscriber("state_action_updates", StateActionUpdates, self.update_callback)
		self.queue = list()
		self.rewards = dict(dict())
		self.rewards_this_iteration = list()
		rospy.Service('state_action_updates', StateActionUpdatesService, self.update_callback)
		self.create_totals = 0
		self.create_totals_list = list()
		self.merge_totals = 0
		self.merge_totals_list = list()
		self.current_iteration_totals = list()
		self.learning_it = 0
		self.rand_val = 0.1
		self.rand_unexplored_val = 0.5
		self.current_transitions = list()
		self.unexplored_transitions = dict(list())
	def get_action(self, state):
		#print self.unexplored_transitions
		if self.additive_states:
			if random.random() < self.rand_unexplored_val and state in self.unexplored_transitions:
				if self.unexplored_transitions[state] != []:
					 transition = random.choice(self.unexplored_transitions[state])
					 if transition['Sampled'] >= 2000:
						action = transition['Action']
						self.unexplored_transitions[state].remove(transition)
						return action
					 transition['Sampled'] += 1
					 return transition['Action']
		if random.random() < self.rand_val:
			if self.additive_states:
				return random.choice(self.q_table_additive_states[state].keys())
			else:
				return random.choice(self.q_table[state].keys())
		if self.additive_states:
			current_max = -9999
			best_action = list()
			for action, temp in self.q_table_additive_states[state].iteritems():
				for val in temp:
					if val['V'] > current_max:
						best_action = list()
						best_action.append(action)
						current_max = val['V']
					if val['V'] == current_max:
						best_action.append(action)
			return random.choice(best_action)		
		else:
			val = max(self.q_table[state].iteritems(), key=operator.itemgetter(1))
			val2 = [k for k, v, in self.q_table[state].items() if v == val[1]]
			if len(val2) == 1:
				return val2[0]
			else:
				return random.choice(val2)

	def max_action_value(self, state):
		if state not in self.q_table:
			return 0
		val = max(self.q_table[state].iteritems(), key=operator.itemgetter(1))
		return val[1]

	def max_action_value_list(self, state):
		if state not in self.q_table:
			return 0
		current_max = -9999
		for temp in self.q_table_additive_states[state].itervalues():
			for val in temp:
				if val['V'] > current_max:
					current_max = val['V']
		return current_max

	def update(self, next_state_update, action_update, state_update, reward):
		#smach.logerr('Next-state_update: ' + next_state_update + ' state_update: ' + state_update + ' action_update: ' + action_update + ' Reward:' + str(reward))
		if self.q_learning and not self.additive_states:
			self.q_table[state_update][action_update] = self.q_table[state_update][action_update] + self.learning_rate * (reward + self.discount_factor * self.max_action_value(next_state_update) - self.q_table[state_update][action_update])

		if self.additive_states:
			if self.q_table_additive_states[state_update][action_update][0]['V'] == 0:
				self.q_table_additive_states[state_update][action_update][0] = {'V':reward, 'mean':reward, 'std':np.abs(1)}
			update_ele_list = self.most_similar_gaussian(state_update, action_update, reward)

			self.current_transitions.append((state_update, action_update))

			done = False
			for i in range(0, len(self.q_table_additive_states[state_update][action_update])):
				for j in range(i, len(self.q_table_additive_states[state_update][action_update])):
					if i == j:
						continue
					val1 = self.q_table_additive_states[state_update][action_update][i]
					val2 = self.q_table_additive_states[state_update][action_update][j]
					overlap = self.check_gaussian_overlap(val1['mean'], val1['std'], val2['mean'], val2['std'])
					if overlap[1] > .01:
						self.merge_totals += 1
						
						self.rewards[state_update][action_update][i] = self.rewards[state_update][action_update][i] + self.rewards[state_update][action_update][j]
						self.additive_states_prob_counts[state_update][action_update][i] = self.additive_states_prob_counts[state_update][action_update][i] + self.additive_states_prob_counts[state_update][action_update][i]
						total = sum(self.additive_states_prob_counts[state_update][action_update])
						for gaus in range(0, len(self.additive_states_prob[state_update][action_update])):
							self.additive_states_prob[state_update][action_update][gaus] = self.additive_states_prob_counts[state_update][action_update][gaus]/total						
						self.fit_gaussian(state_update, action_update, i)
						
						self.additive_states_prob[state_update][action_update].pop(j)
						self.additive_states_prob_counts[state_update][action_update].pop(j)
						self.rewards[state_update][action_update].pop(j)
						for state in self.unexplored_transitions.keys():
							for transition_it in self.unexplored_transitions[state]:
								if transition_it['transitions'] == self.q_table_additive_states[state_update][action_update][j]:
									self.unexplored_transitions[state].remove(transition_it)

						self.q_table_additive_states[state_update][action_update].pop(j)
						update_ele_list = self.most_similar_gaussian(state_update, action_update, reward)
						done = True
						break
				if done:
					break


			for (update_ele, prob) in update_ele_list:
				if prob > 4:
					self.create_totals += 1
					update_ele = self.add_state_action_gaussian(state_update, action_update, reward)

				try:
					self.rewards[state_update][action_update][update_ele].append(reward)
				except:
					print state_update
					print action_update
					print update_ele
					print self.rewards
					time.sleep(10000)
				self.fit_gaussian(state_update, action_update, update_ele)


				self.additive_states_prob_counts[state_update][action_update][update_ele] += 1
				total = sum(self.additive_states_prob_counts[state_update][action_update])
				for gaus in range(0, len(self.additive_states_prob[state_update][action_update])):
					self.additive_states_prob[state_update][action_update][gaus] = self.additive_states_prob_counts[state_update][action_update][gaus]/total
		
				self.q_table_additive_states[state_update][action_update][update_ele]['V'] = self.q_table_additive_states[state_update][action_update][update_ele]['V'] + self.learning_rate * (reward + self.discount_factor * self.max_action_value_list(next_state_update) - self.q_table_additive_states[state_update][action_update][update_ele]['V'])


		if self.learning_it == 99995 and self.once == False:
			print self.unexplored_transitions
			print self.q_table_additive_states
			print self.additive_states_prob
			print self.additive_states_prob_counts

			#cur_time = str(int(time.time()))
			#target = open('/nfs/attic/smartw/users/curranw/smach_rl/QTable_totals' + curTime + '.txt','a')
			#target.write(str(self.q_table_additive_states))
			#target.write(str(self.q_table))
			#target = open('/nfs/attic/smartw/users/curranw/smach_rl/Prob_totals' + curTime + '.txt','a')
			#target.write(str(self.additive_states_prob))
			#write('/nfs/attic/smartw/users/curranw/smach_rl/create_totals' + curTime + '.csv', self.current_iteration_totals, self.create_totals_list)
			#time.sleep(10)
			#write('/nfs/attic/smartw/users/curranw/smach_rl/merge_totals' + curTime + '.csv', self.current_iteration_totals, self.merge_totals_list)
			#time.sleep(10)
			self.once = True

	def add_state_action_gaussian(self, state, action, reward):
		#self.q_table_additive_states[state][action].append({'V':reward, 'mean':reward, 'std':np.abs(reward/4.0)})
		self.q_table_additive_states[state][action].append({'V':reward, 'mean':reward, 'std':np.abs(1)})
		self.additive_states_prob[state][action].append(1.0)
		self.additive_states_prob_counts[state][action].append(1.0)
		self.rewards[state][action].append([])

		update_ele = len(self.q_table_additive_states[state][action]) - 1
		for (cur_transition_state, cur_transition_action) in self.current_transitions:
			if cur_transition_state not in self.unexplored_transitions.keys():
				self.unexplored_transitions[cur_transition_state] = list()
			self.unexplored_transitions[cur_transition_state].append({'Action': cur_transition_action , 'Sampled': 0, 'transitions':self.q_table_additive_states[state][action][update_ele]})
			#print self.unexplored_transitions
			#time.sleep(20)

		return len(self.rewards[state][action])-1

	def fit_gaussian(self, state, action, update_ele):
		reward_list = self.rewards[state][action][update_ele]
		if len(reward_list) < 5:
			self.q_table_additive_states[state][action][update_ele]['std'] = np.abs(1)
			self.q_table_additive_states[state][action][update_ele]['mean'] = self.rewards[state][action][update_ele][0]
			return
		r_mean = np.mean(reward_list)
		r_std = np.std(reward_list)
		std = self.q_table_additive_states[state][action][update_ele]['std']
		mean = self.q_table_additive_states[state][action][update_ele]['mean']
		self.q_table_additive_states[state][action][update_ele]['std'] += 1 * (r_std - std)
		self.q_table_additive_states[state][action][update_ele]['mean'] += 1 * (r_mean - mean)
	def most_similar_gaussian(self, state, action, reward):
		update_ele = list()
		ele_it = 0
		min_ele = (-1, 9999)
		for ele in self.q_table_additive_states[state][action]:
			prob = np.abs((ele['mean'] - reward)/ele['std'])
			if prob <= 4:
				update_ele.append((ele_it, prob))
			else:
				if prob < min_ele[1]:
					min_ele = (ele_it, prob)
			ele_it += 1

		if update_ele == []:
			update_ele.append(min_ele)
		return update_ele

	def add_state_actions(self, state, action_list):
		if state not in self.q_table.keys():
			self.q_table[state] = dict()
			self.q_table_additive_states[state] = dict()
			self.additive_states_prob[state] = dict()
			self.additive_states_prob_counts[state] = dict()
		for action in action_list:
			if action not in self.q_table_additive_states[state].keys():
				self.q_table_additive_states[state][action] = list()
				self.additive_states_prob[state][action] = list()
				self.additive_states_prob_counts[state][action] = list()
			#self.q_table[state][action] = random.random()
			self.q_table[state][action] = 0
			self.q_table_additive_states[state][action].append({'V':0, 'mean':0, 'std':1})
			self.additive_states_prob[state][action].append(1.0)
			self.additive_states_prob_counts[state][action].append(1.0)
			if state not in self.rewards.keys():
				self.rewards[state] = dict()
			self.rewards[state][action] = list()
			self.rewards[state][action].append([])


	def queue_state_update(self, state, action):
		#smach.logerr('Added to Queue: ' + 'State: ' + state + ' Action: ' + action)
		self.states_to_update.append((state, action))

	def update_queued_states(self, reward):
		if self.q_learning and not self.additive_states:
			for state, action in self.states_to_update:
				self.q_table[state][action] = self.q_table[state][action] + self.learning_rate * (reward - self.q_table[state][action])
		if self.additive_states:
			for state, action in self.states_to_update:
				#smach.logerr('Queued: State: ' + state + ' Action: ' + action + ' Reward:' + str(reward))
				self.current_transitions.append((state, action))
				update_ele_list = self.most_similar_gaussian(state, action, reward)
				
				#for (update_ele, prob) in update_ele_list:
				for update_ele in [temp[0] for temp in update_ele_list]:
					self.q_table_additive_states[state][action][update_ele]['V'] = self.q_table_additive_states[state][action][update_ele]['V'] + self.learning_rate * (reward - self.q_table_additive_states[state][action][update_ele]['V'])


		self.states_to_update = list()

	def update_callback(self, msg):
		#print self.q_table_additive_states
		self.current_iteration += 1
		if self.current_iteration > 0:		
			if self.q_learning:
				remove_ele = None
				for ele in self.queue:
					if ele.container_num == msg.container_num:
						if msg.state == msg.action and msg.state != "END":
							ele.reward += msg.reward
							return
						self.update(msg.state, ele.action, ele.state, ele.reward)
						remove_ele = ele
				if remove_ele != None:
					self.queue.remove(remove_ele)

				self.update_queued_states(msg.reward)
				if msg.state != "END":
					self.queue.append(msg)

		if msg.state == "END":
			self.learning_it += 1
			self.current_iteration_totals.append(self.learning_it)
			self.merge_totals_list.append(self.merge_totals)
			self.create_totals_list.append(self.create_totals)
			#self.rand_val = self.rand_val * .999893
			self.current_transitions = []
			if self.learning_it % 100 == 0:
				rospy.logerr(self.learning_it)

	def check_gaussian_overlap(self, mean1, sd1, mean2, sd2):
		gauss_one = np.random.normal(mean1, sd1, 1000)
		gauss_two = np.random.normal(mean2, sd2, 1000)
		val = stats.ttest_ind(gauss_one, gauss_two)
		return val

def write(path, y_list, x_list):
	target = open(path,'a')
	for it in range(0, len(x_list)):
		target.write(str(y_list[it]) + ',' + str(x_list[it]) + '\n')

	target.close()

import os
def save(path, ext='png', close=True, verbose=True):
	"""Save a figure from pyplot.
 
	Parameters
	----------
	path : string
		The path (and filename, without the extension) to save the
		figure to.
 
	ext : string (default='png')
		The file extension. This must be supported by the active
		matplotlib backend (see matplotlib.backends module).  Most
		backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
 
	close : boolean (default=True)
		Whether to close the figure after saving.  If you want to save
		the figure multiple times (e.g., to multiple formats), you
		should NOT close it in between saves or you will have to
		re-plot it.
 
	verbose : boolean (default=True)
		Whether to print information about when and where the image
		has been saved.
 
	"""
	
	# Extract the directory and filename from the given path
	directory = os.path.split(path)[0]
	filename = "%s.%s" % (os.path.split(path)[1], ext)
	if directory == '':
		directory = '.'
 
	# If the directory does not exist, create it
	if not os.path.exists(directory):
		os.makedirs(directory)
 
	# The final path to save to
	savepath = os.path.join(directory, filename)
 
	if verbose:
		print("Saving figure to '%s'..." % savepath),
 
	# Actually save the figure
	pylab.savefig(savepath)
	
	# Close it
	if close:
		pylab.close()
 
	if verbose:
		print("Done")
