#!/usr/bin/env python
import time
import operator
import random
import rospy
import smach
from smach_msgs.msg import StateActionUpdates
import pylab
import numpy as np
from scipy import interpolate
from scipy import stats
from scipy import integrate
from smach_msgs.srv import StateActionUpdatesService
from sklearn import mixture
import math


class QLearner():
	def __init__(self):
		self.once = False
		self.q_learning = True
		self.additive_states = True
		self.it = 0

		self.QTable = dict(dict())
		self.QTable_additive_states = dict(dict())
		self.additive_states_probabilities = dict(dict())
		self.additive_states_probabilities_counts = dict(dict())

		self.learning_rate = 0.1
		self.discount_factor = 0.9
		self.states_to_update = list()
		rospy.Subscriber("state_action_updates", StateActionUpdates, self.update_callback)
		self.queue = list()
		self.rewards = dict(dict())
		self.rewards_this_iteration = list()
		s = rospy.Service('state_action_updates', StateActionUpdatesService, self.update_callback)
		self.create_totals = 0
		self.create_totals_list = list()
		self.merge_totals = 0
		self.merge_totals_list = list()
		self.it_totals = list()
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
					 v = random.choice(self.unexplored_transitions[state])
					 if v['Sampled'] >= 2000:
					 	action = v['Action']
					 	self.unexplored_transitions[state].remove(v)
					 	return action
					 v['Sampled'] += 1
					 print v
					 #time.sleep(20)
					 return v['Action']
		if random.random() < self.rand_val:
			if self.additive_states:
				return random.choice(self.QTable_additive_states[state].keys())
			else:
				return random.choice(self.QTable[state].keys())
		if self.additive_states:
			current_max = -9999
			best_action = list()
			for action,temp in self.QTable_additive_states[state].iteritems():
				for val in temp:
					if val['V'] > current_max:
						best_action = list()
						best_action.append(action)
						current_max = val['V']
					if val['V'] == current_max:
						best_action.append(action)
			return random.choice(best_action)		
		else:
			val = max(self.QTable[state].iteritems(), key=operator.itemgetter(1))
			val2 = [k for k,v, in self.QTable[state].items() if v == val[1]]
			if len(val2) == 1:
				return val2[0]
			else:
				return random.choice(val2)

	def max_action_value(self, state):
		if state not in self.QTable:
			#smach.logerr('%s not found in QTable', state)
			return 0
		val = max(self.QTable[state].iteritems(), key=operator.itemgetter(1))
		return val[1]

	def max_action_value_list(self, state):
		if state not in self.QTable:
			#smach.logerr('%s not found in QTable', state)
			return 0
		current_max = -9999
		for action,temp in self.QTable_additive_states[state].iteritems():
			for val in temp:
				if val['V'] > current_max:
					current_max = val['V']
		return current_max

	def update(self, next_state, action, state, reward):
		#smach.logerr('Next-State: ' + next_state + ' State: ' + state + ' Action: ' + action + ' Reward:' + str(reward))
		if self.q_learning and not self.additive_states:
			self.QTable[state][action] = self.QTable[state][action] + self.learning_rate * (reward + self.discount_factor * self.max_action_value(next_state) - self.QTable[state][action])

		if self.additive_states:
			if self.QTable_additive_states[state][action][0]['V'] == 0:
				self.QTable_additive_states[state][action][0] = {'V':reward, 'mean':reward, 'std':np.abs(1)}
			update_ele_list = self.most_similar_gaussian(state, action, reward)

			self.current_transitions.append((state,action))

			done = False
			for i in range(0, len(self.QTable_additive_states[state][action])):
				for j in range(i, len(self.QTable_additive_states[state][action])):
					if i == j:
						continue
					val1 = self.QTable_additive_states[state][action][i]
					val2 = self.QTable_additive_states[state][action][j]
					overlap = self.check_gaussian_overlap(val1['mean'], val1['std'], val2['mean'], val2['std'])
					if overlap[1] > .01:
						self.merge_totals += 1
						
						self.rewards[state][action][i] = self.rewards[state][action][i] + self.rewards[state][action][j]
						self.additive_states_probabilities_counts[state][action][i] = self.additive_states_probabilities_counts[state][action][i] + self.additive_states_probabilities_counts[state][action][i]
						total = sum(self.additive_states_probabilities_counts[state][action])
						for it in range(0, len(self.additive_states_probabilities[state][action])):
							self.additive_states_probabilities[state][action][it] = self.additive_states_probabilities_counts[state][action][it]/total						
						self.fit_gaussian(state, action, i)
						
						self.additive_states_probabilities[state][action].pop(j)
						self.additive_states_probabilities_counts[state][action].pop(j)
						self.rewards[state][action].pop(j)
						for s in self.unexplored_transitions.keys():
							for t in self.unexplored_transitions[s]:
								if t['transitions'] == self.QTable_additive_states[state][action][j]:
									self.unexplored_transitions[s].remove(t)

						self.QTable_additive_states[state][action].pop(j)
						update_ele_list = self.most_similar_gaussian(state, action, reward)
						done = True
						break
				if done:
					break


			for (update_ele, prob) in update_ele_list:
				if prob > 4:
					self.create_totals += 1
					update_ele = self.add_state_action_gaussian(state, action, reward)

				try:
					self.rewards[state][action][update_ele].append(reward)
				except:
					print state
					print action
					print update_ele
					print self.rewards
					time.sleep(10000)
				self.fit_gaussian(state, action, update_ele)


				self.additive_states_probabilities_counts[state][action][update_ele] += 1
				total = sum(self.additive_states_probabilities_counts[state][action])
				for it in range(0, len(self.additive_states_probabilities[state][action])):
					self.additive_states_probabilities[state][action][it] = self.additive_states_probabilities_counts[state][action][it]/total
		
				self.QTable_additive_states[state][action][update_ele]['V'] = self.QTable_additive_states[state][action][update_ele]['V'] + self.learning_rate * (reward + self.discount_factor * self.max_action_value_list(next_state) - self.QTable_additive_states[state][action][update_ele]['V'])


		if self.learning_it == 99995 and self.once == False:
			print self.unexplored_transitions
			print self.QTable_additive_states
			print self.additive_states_probabilities
			print self.additive_states_probabilities_counts
			tot = 0
			tot_2 = 0
			for s,rest in self.QTable_additive_states.iteritems():
				for a,l in self.QTable_additive_states[s].iteritems():
					tot += len(l)
			for s,l in self.unexplored_transitions.iteritems():
				tot_2 += len(l)

			print tot
			print tot_2
			curTime = str(int(time.time()))
			#target = open('/nfs/attic/smartw/users/curranw/smach_rl/QTable_totals' + curTime + '.txt','a')
			#target.write(str(self.QTable_additive_states))
			#target.write(str(self.QTable))
			#target = open('/nfs/attic/smartw/users/curranw/smach_rl/Prob_totals' + curTime + '.txt','a')
			#target.write(str(self.additive_states_probabilities))
			#write('/nfs/attic/smartw/users/curranw/smach_rl/create_totals' + curTime + '.csv', self.it_totals, self.create_totals_list)
			#time.sleep(10)
			#write('/nfs/attic/smartw/users/curranw/smach_rl/merge_totals' + curTime + '.csv', self.it_totals, self.merge_totals_list)
			#time.sleep(10)
			self.once = True

	def add_state_action_gaussian(self, state, action, reward):
		#self.QTable_additive_states[state][action].append({'V':reward, 'mean':reward, 'std':np.abs(reward/4.0)})
		self.QTable_additive_states[state][action].append({'V':reward, 'mean':reward, 'std':np.abs(1)})
		self.additive_states_probabilities[state][action].append(1.0)
		self.additive_states_probabilities_counts[state][action].append(1.0)
		self.rewards[state][action].append([])

		update_ele = len(self.QTable_additive_states[state][action]) - 1
		for (s,a) in self.current_transitions:
			if s not in self.unexplored_transitions.keys():
				self.unexplored_transitions[s] = list()
			self.unexplored_transitions[s].append({'Action': a , 'Sampled': 0, 'transitions':self.QTable_additive_states[state][action][update_ele]})
			#print self.unexplored_transitions
			#time.sleep(20)

		return len(self.rewards[state][action])-1

	def fit_gaussian(self, state, action, update_ele):
		r = self.rewards[state][action][update_ele]
		if len(r) < 5:
			self.QTable_additive_states[state][action][update_ele]['std'] = np.abs(1)
			self.QTable_additive_states[state][action][update_ele]['mean'] = self.rewards[state][action][update_ele][0]
			return
		r_mean = np.mean(r)
		r_std = np.std(r)
		std = self.QTable_additive_states[state][action][update_ele]['std']
		mean = self.QTable_additive_states[state][action][update_ele]['mean']
		self.QTable_additive_states[state][action][update_ele]['std'] += 1 * (r_std - std)
		self.QTable_additive_states[state][action][update_ele]['mean'] += 1 * (r_mean - mean)
	def most_similar_gaussian(self, state, action, reward):
		update_ele = list()
		it = 0
		min_ele = (-1, 9999)
		for ele in self.QTable_additive_states[state][action]:
			prob = np.abs((ele['mean'] - reward)/ele['std'])
			if prob <= 4:
				update_ele.append((it,prob))
			else:
				if prob < min_ele[1]:
					min_ele = (it, prob)
			it += 1

		if update_ele == []:
			update_ele.append(min_ele)
		return update_ele

	def add_state_actions(self, state, action):
		if state not in self.QTable.keys():
			self.QTable[state] = dict()
			self.QTable_additive_states[state] = dict()
			self.additive_states_probabilities[state] = dict()
			self.additive_states_probabilities_counts[state] = dict()
		for a in action:
			if a not in self.QTable_additive_states[state].keys():
				self.QTable_additive_states[state][a] = list()
				self.additive_states_probabilities[state][a] = list()
				self.additive_states_probabilities_counts[state][a] = list()
			#self.QTable[state][a] = random.random()
			self.QTable[state][a] = 0
			self.QTable_additive_states[state][a].append({'V':0, 'mean':0, 'std':1})
			self.additive_states_probabilities[state][a].append(1.0)
			self.additive_states_probabilities_counts[state][a].append(1.0)
			if state not in self.rewards.keys():
				self.rewards[state] = dict()
			self.rewards[state][a] = list()
			self.rewards[state][a].append([])


	def queue_state_update(self, state, action):
		#smach.logerr('Added to Queue: ' + 'State: ' + state + ' Action: ' + action)
		self.states_to_update.append((state,action))

	def update_queued_states(self, next_state, reward):
		if self.q_learning and not self.additive_states:
			for state,action in self.states_to_update:
				self.QTable[state][action] = self.QTable[state][action] + self.learning_rate * (reward - self.QTable[state][action])
		if self.additive_states:
			for state,action in self.states_to_update:
				#smach.logerr('Queued: State: ' + state + ' Action: ' + action + ' Reward:' + str(reward))
				self.current_transitions.append((state,action))
				update_ele_list = self.most_similar_gaussian(state, action, reward)
				for (update_ele, prob) in update_ele_list:
					self.QTable_additive_states[state][action][update_ele]['V'] = self.QTable_additive_states[state][action][update_ele]['V'] + self.learning_rate * (reward - self.QTable_additive_states[state][action][update_ele]['V'])


		self.states_to_update = list()

	def update_callback(self, msg):
		#print self.QTable_additive_states
		self.it += 1
		if self.it > 0:		
			if self.q_learning:
				removeEle = None
				for ele in self.queue:
					if ele.container_num == msg.container_num:
						if msg.state == msg.action and msg.state != "END":
							ele.reward += msg.reward
							return
						self.update(msg.state, ele.action, ele.state, ele.reward)
						removeEle = ele
				if removeEle != None:
					self.queue.remove(removeEle)

				self.update_queued_states(msg.state, msg.reward)
				if msg.state != "END":
					self.queue.append(msg)

		if msg.state == "END":
			self.learning_it += 1
			self.it_totals.append(self.learning_it)
			self.merge_totals_list.append(self.merge_totals)
			self.create_totals_list.append(self.create_totals)
			#self.rand_val = self.rand_val * .999893
			self.current_transitions = []
			if self.learning_it % 100 == 0:
				smach.logerr(self.learning_it)

	def check_gaussian_overlap(self, mean1, sd1, mean2, sd2):
		a = np.random.normal(mean1, sd1, 1000)
		b = np.random.normal(mean2, sd2, 1000)
		val = stats.ttest_ind(a, b)
		return val

def write(path, y, x):
	target = open(path,'a')
	for it in range(0, len(x)):
		target.write(str(y[it]) + ',' + str(x[it]) + '\n')

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
