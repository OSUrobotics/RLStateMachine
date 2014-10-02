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



#Average time it takes to perfectly navigate to the goal.
navigation_time = 60

#Average time time it takes to perfectly perform the task.
task_time = 60


nav_teleop_variance = 10
nav_teleop_mean = 60
nav_shared_variance = 5
nav_shared_mean = 50
nav_automa_variance = 2
nav_automa_mean = 50
nav_automa_failure  = 0.05
nav_automa_failure_scale = 2
task_teleop_variance = 5
task_teleop_mean = 20
task_automa_variance = 2
task_automa_mean = 10
pack_arms_mean = 14.9
pack_arms_variance = .2455
unpack_arms_mean = 9.7859
unpack_arms_variance = .2085

class QLearner():
	def __init__(self):
		self.once = False
		self.q_learning = True
		self.gaussian = False
		self.multi_mdp = False
		self.additive_states = False
		self.cheating = True
		self.it = 0
		self.polynomials = dict(dict())

		self.QTable = dict(dict())
		self.QTable_additive_states = dict(dict())
		self.QTable_risk = dict(dict())
		self.additive_states_probabilities = dict(dict())
		self.additive_states_probabilities_counts = dict(dict())
		self.QTable_gauss = dict(dict())
		self.QTable_multi_mdp = list(dict(dict()))

		self.learning_rate = 0.1
		self.discount_factor = 0.9
		self.states_to_update = list()
		rospy.Subscriber("state_action_updates", StateActionUpdates, self.update_callback)
		self.queue = list()
		self.rewards = dict(dict())
		self.rewards_this_iteration = list()
		self.rewards_gauss = dict(dict())
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
					 if v['Sampled'] >= 4000:
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
		if self.gaussian:
			best_mean = -99999
			best_action = list()
			for (action,gmm) in self.QTable_gauss[state].iteritems():
				for mean in gmm.means_:
					if mean > best_mean:
						best_action = list()
						best_action.append(action)
						best_mean = mean
					if mean == best_mean:
						best_action.append(action)
			return random.choice(best_action)
		elif self.additive_states:
			current_max = -9999
			best_action = list()
			for action,temp in self.QTable_additive_states[state].iteritems():
				for val in temp:
					print val
					if val['V'] > current_max:
						best_action = list()
						best_action.append(action)
						current_max = val['V']
					if val['V'] == current_max:
						best_action.append(action)
			print best_action
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
		print current_max
		return current_max

	def max_action_value_gauss(self, state):
		if state not in self.QTable:
			#smach.logerr('%s not found in QTable', state)
			temp = [0,0,0,0,0]
			gmm = mixture.gmm.GMM(n_components=1)
			gmm.fit(temp)
			return gmm
		best_mean = -99999
		best_action = 0
		for (action,gmm) in self.QTable_gauss[state].iteritems():
			for mean in gmm.means_:
				if mean > best_mean:
					best_action = gmm
					best_mean = mean
		return best_action
		#val = max(self.QTable[state].iteritems(), key=operator.itemgetter(1))
		#return val[1]
	def update(self, next_state, action, state, reward):
		print self.QTable
		#smach.logerr('Next-State: ' + next_state + ' State: ' + state + ' Action: ' + action + ' Reward:' + str(reward))
		if self.q_learning and not self.gaussian and not self.multi_mdp and not self.additive_states:
			print 'GOT HERE! NOOOOO'
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
					#if overlap[0] > .50:
					if overlap[1] > .01:
						self.merge_totals += 1
						#print 'OVERLAP HIGH'
						#print overlap
						#time.sleep(2)
						
						self.rewards[state][action][i] = self.rewards[state][action][i] + self.rewards[state][action][j]
						self.additive_states_probabilities_counts[state][action][i] = self.additive_states_probabilities_counts[state][action][i] + self.additive_states_probabilities_counts[state][action][i]
						total = sum(self.additive_states_probabilities_counts[state][action])
						for it in range(0, len(self.additive_states_probabilities[state][action])):
							self.additive_states_probabilities[state][action][it] = self.additive_states_probabilities_counts[state][action][it]/total						
						self.fit_gaussian(state, action, i)
						
						self.additive_states_probabilities[state][action].pop(j)
						self.additive_states_probabilities_counts[state][action].pop(j)
						self.rewards[state][action].pop(j)
						#time.sleep(20)
						for s in self.unexplored_transitions.keys():
							for t in self.unexplored_transitions[s]:
								#print t
								#print s
								#print self.unexplored_transitions
								if t['transitions'] == self.QTable_additive_states[state][action][j]:
									self.unexplored_transitions[s].remove(t)
									#time.sleep(20)

						self.QTable_additive_states[state][action].pop(j)
						update_ele_list = self.most_similar_gaussian(state, action, reward)
						done = True
						#print self.QTable_additive_states
						#time.sleep(20)
						break
				if done:
					break


			for (update_ele, prob) in update_ele_list:
				if prob > 4:
					self.create_totals += 1
					#time.sleep(2)
					update_ele = self.add_state_action_gaussian(state, action, reward)

				#print self.rewards
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
			time.sleep(10)
			target = open('/nfs/attic/smartw/users/curranw/smach_rl/QTable_totals' + curTime + '.txt','a')
			#target.write(str(self.QTable_additive_states))
			target.write(str(self.QTable))
			#target = open('/nfs/attic/smartw/users/curranw/smach_rl/Prob_totals' + curTime + '.txt','a')
			#target.write(str(self.additive_states_probabilities))
			#write('/nfs/attic/smartw/users/curranw/smach_rl/create_totals' + curTime + '.csv', self.it_totals, self.create_totals_list)
			#time.sleep(10)
			#write('/nfs/attic/smartw/users/curranw/smach_rl/merge_totals' + curTime + '.csv', self.it_totals, self.merge_totals_list)
			#time.sleep(10)
			self.once = True
		if self.gaussian:
			#reward + self.discount_factor * self.max_action_value(next_state)
			next_val = self.max_action_value_gauss(next_state)
			# if isinstance(self.rewards_gauss[state][action],mixture.GMM):
			# 	for gaussian in range(0,len(next_val.means_)):
			# 		next_val.means_[gaussian] *= self.discount_factor
			# 		next_val.covars_[gaussian] *= pow(self.discount_factor,2)
			# 	first_inside_term = self.add_gmm(self.rewards_gauss[state][action], next_val)
			# else:
			# 	for gaussian in range(0,len(next_val.means_)):
			# 		next_val.means_[gaussian] *= self.discount_factor 
			# 		next_val.means_[gaussian] += self.rewards_gauss[state][action]
			# 		next_val.covars_[gaussian] *= pow(self.discount_factor,2)
			# 	first_inside_term = next_val

			if not isinstance(self.rewards_gauss[state][action],mixture.GMM):
				first_inside_term = next_val
			else:
				first_inside_term = self.add_gmm(self.rewards_gauss[state][action], next_val, sample_one=self.learning_rate, sample_two=self.discount_factor*self.learning_rate, add=True)
			#first_inside_term - self.QTable_gauss[state][action]
			second_term = self.subtract_gmm(first_inside_term, self.QTable_gauss[state][action])

			#learning_rate * second_term
			#for gaussian in range(0,len(second_term.means_)):
			#	second_term.means_[gaussian] *= self.learning_rate
			#	second_term.covars_[gaussian] *= pow(self.learning_rate,2)

			#self.QTable[state][action] + second_term
			originalQ = self.QTable_gauss[state][action]
			self.QTable_gauss[state][action] = self.add_gmm(self.QTable_gauss[state][action], second_term)
			
			# if action == "NavAuto" and state == "START_NAV":
			# 	pylab.subplot(6, 1, 1)
			# 	pylab.hist(self.rewards_gauss[state][action].sample(1000), label="reward")
			# 	pylab.legend()
			# 	pylab.subplot(6, 1, 2)
			# 	pylab.hist(next_val.sample(1000), label="next_val")
			# 	pylab.legend()
			# 	pylab.subplot(6, 1, 3)
			# 	pylab.hist(first_inside_term.sample(1000), label="first")
			# 	pylab.legend()
			# 	pylab.subplot(6, 1, 4)
			# 	pylab.hist(second_term.sample(1000), label="second")
			# 	pylab.legend()
			# 	pylab.subplot(6, 1, 5)
			# 	pylab.hist(originalQ.sample(1000), label="original")
			# 	pylab.legend()
			# 	pylab.subplot(6, 1, 6)
			# 	pylab.hist(self.QTable_gauss[state][action].sample(1000), label="result")
			# 	pylab.legend()
			# 	pylab.show()


			#smach.logerr(self.it)
			if self.it % 10 == 0:
				gmm1 = self.QTable_gauss["START_NAV"]["NavTele"]
				gmm2 = self.QTable_gauss["START_NAV"]["NavAuto"]
				data_gmm1 = list()
				data_gmm2 = list()
				for gaussian in range(0,len(gmm1.means_)):
					temp = np.sqrt(gmm1.covars_[gaussian]) * np.random.randn(gmm1.weights_[gaussian]*10000,1) + gmm1.means_[gaussian]
					data_gmm1 = data_gmm1 + np.concatenate(temp).tolist()
				pylab.hist(data_gmm1, bins=100, label="Tele")
				
				for gaussian in range(0,len(gmm2.means_)):
					temp = np.sqrt(gmm2.covars_[gaussian]) * np.random.randn(gmm2.weights_[gaussian]*10000,1) + gmm2.means_[gaussian]
					data_gmm2 = data_gmm2 + np.concatenate(temp).tolist()
				pylab.hist(data_gmm2, bins=100, label="Auto", color='y')

				#smach.logerr(self.QTable_gauss["START_NAV"]["NavTele"].means_)
				#smach.logerr(self.QTable_gauss["START_NAV"]["NavTele"].covars_)
				#smach.logerr(self.QTable_gauss["START_NAV"]["NavAuto"].means_)
				#smach.logerr(self.QTable_gauss["START_NAV"]["NavAuto"].covars_)
				pylab.legend()
				pylab.ylim([0,1000])
				pylab.xlim([-150,0])
				save("/nfs/attic/smartw/users/curranw/smach_rl/" + str(self.it))
				#pylab.show()
	

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
			#self.QTable_additive_states[state][action][update_ele]['std'] = np.abs(reward/4.0)
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
			#first_term_denom = ele['std'] * math.sqrt(2 * math.pi)
			#second_term_num = math.pow((reward - ele['mean']),2)
			#second_term_denom = 2 * math.pow(ele['std'],2)
			#second_term = math.exp(- (second_term_num/second_term_denom))
			#prob = 1.0/first_term_denom * second_term
			prob = np.abs((ele['mean'] - reward)/ele['std'])
			#print prob
			#time.sleep(1)
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
			self.QTable_gauss[state] = dict()
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
			if self.cheating:
				self.add_state_actions_cheat(state,a)
			else:
				self.QTable_additive_states[state][a].append((0,0,0))			
			temp = [0,0,0,0,0]
			self.QTable_gauss[state][a] = mixture.gmm.GMM(n_components=1)
			self.QTable_gauss[state][a].fit(temp)

	def queue_state_update(self, state, action):
		#smach.logerr('Added to Queue: ' + 'State: ' + state + ' Action: ' + action)
		self.states_to_update.append((state,action))

	def update_queued_states(self, next_state, reward):
		if self.q_learning and not self.gaussian and not self.multi_mdp and not self.additive_states:
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

		#if msg.state != "END":
			#self.store_rewards(msg.reward, msg.state, msg.action)
		#if msg.state == "END":
			#self.update_gaussians()

		if msg.state == "END":
			self.learning_it += 1
			self.it_totals.append(self.learning_it)
			self.merge_totals_list.append(self.merge_totals)
			self.create_totals_list.append(self.create_totals)
			#self.rand_val = self.rand_val * .999893
			self.current_transitions = []
			if self.learning_it % 100 == 0:
				smach.logerr(self.learning_it)
		if self.multi_mdp and self.it == 200:
			self.build_multi_mdp()

	def build_multi_mdp(self):
		totals = list()
		for s in self.rewards_gauss:
			for a in self.rewards_gauss[s]:
				totals.append((s,a,len(self.rewards_gauss[s][a].means_)))
		total = 1
		print totals
		time.sleep(100)
		for (s,a,v) in totals:
			total *= v
		print total
		time.sleep(100)
		for it in range(0, total):
			table = dict(dict())
			for s in self.rewards_gauss:
				table[s] = dict()
				for a,gmm in self.rewards_gauss[s].iteritems():
					table[s][a] = (0, gmm)
			self.QTable_multi_mdp.append(table)
		time.sleep(100)

	def store_rewards(self, reward, state, action):
		if state not in self.rewards.keys():
			self.rewards[state] = dict()
		if action not in self.rewards[state].keys():
			self.rewards[state][action] = list()

		self.rewards[state][action].append(reward)
		self.rewards_this_iteration.append((state,action))

	def check_gaussian_overlap(self, mean1, sd1, mean2, sd2):
		a = np.random.normal(mean1, sd1, 1000)
		b = np.random.normal(mean2, sd2, 1000)
		#kde_a = stats.gaussian_kde(a)
		#kde_b = stats.gaussian_kde(b)
		#def y_pts(pt):
		#	y_pt = min(kde_a(pt), kde_b(pt))
		#	return y_pt

		#overlap = integrate.quad(y_pts, np.min((a,b)), np.max((a,b)))
		val = stats.ttest_ind(a, b)
		#print "t-test!"
		#print val
		return val
		#return overlap

	def update_gaussians(self):
		self.it += 1

		for (state,action) in self.rewards_this_iteration:
			#action = "NavTele"
			#smach.logerr(state + ',' + action)

			r1 = self.rewards[state][action]
			if len(r1) < 3:
				#smach.logerr('r1 < 3')
				continue
			
			(poly1, poly1d, poly1dd) = self.get_splines(r1)

			num_gaussians = self.find_num_gaussians(poly1, poly1d, poly1dd)

			if num_gaussians == 0:
				#smach.logerr('zero gaussians')
				#smach.logerr(r1)
				if state not in self.rewards_gauss.keys():
					self.rewards_gauss[state] = dict()
				self.rewards_gauss[state][action] = np.mean(r1)
				continue
			gmm = mixture.gmm.GMM(n_components=num_gaussians)
			gmm.fit(r1)
			data = list()
#				for gaussian in range(0,num_gaussians):
#					temp = np.sqrt(gmm.covars_[gaussian]) * np.random.randn(gmm.weights_[gaussian]*1000,1) + gmm.means_[gaussian]
#					if state not in self.rewards_gauss.keys():
#						self.rewards_gauss[state] = dict()
				#self.rewards_gauss[state][action] = (gmm.means_[gaussian], gmm.covars_[gaussian], gmm.weights_[gaussian])
			if state not in self.rewards_gauss.keys():
				self.rewards_gauss[state] = dict()

			self.rewards_gauss[state][action] = gmm

		if self.it % 1000000 == 0 and self.it != 0:
			action = "NavTele"
			r1 = self.rewards["START_NAV"][action]
			(hist1, bins1) = np.histogram(r1, bins=20, range=(min(r1)-np.std(r1),max(r1)+np.std(r1)))
			xx1 = np.linspace(min(r1)-np.std(r1), max(r1)+np.std(r1))
			poly1 = interpolate.InterpolatedUnivariateSpline(bins1[0:len(bins1)-1], hist1)
			poly1d = poly1.derivative()
			poly1d = interpolate.InterpolatedUnivariateSpline(xx1, poly1d(xx1))
			poly1dd = poly1d.derivative()
			poly1dd = interpolate.InterpolatedUnivariateSpline(xx1, poly1dd(xx1))

			num_gaussians = self.find_num_gaussians(poly1, poly1d, poly1dd)
			gmm = mixture.gmm.GMM(n_components=num_gaussians)
			gmm.fit(r1)
			data = list()
			for gaussian in range(0,num_gaussians):
				temp = np.sqrt(gmm.covars_[gaussian]) * np.random.randn(gmm.weights_[gaussian]*1000,1) + gmm.means_[gaussian]
				data = data + np.concatenate(temp).tolist()

			(hist1, bins1) = np.histogram(data, bins=20, range=(min(data)-np.std(data),max(data)+np.std(data)))
			xx1 = np.linspace(min(data)-np.std(data), max(data)+np.std(data))
			#pylab.hist(data, bins=100)
			#pylab.show()			
			pylab.hist(data, bins=20)

			action = "NavAuto"
			r2 = self.rewards["START_NAV"][action]
			(hist2, bins2) = np.histogram(r2, bins=20, range=(min(r2)-np.std(r2),max(r2)+np.std(r2)))
			xx1 = np.linspace(min(r2)-np.std(r2), max(r2)+np.std(r2))
			poly1 = interpolate.InterpolatedUnivariateSpline(bins2[0:len(bins2)-1], hist2)
			poly1d = poly1.derivative()
			poly1d = interpolate.InterpolatedUnivariateSpline(xx1, poly1d(xx1))
			poly1dd = poly1d.derivative()
			poly1dd = interpolate.InterpolatedUnivariateSpline(xx1, poly1dd(xx1))

			num_gaussians = self.find_num_gaussians(poly1, poly1d, poly1dd)
			gmm = mixture.gmm.GMM(n_components=num_gaussians)
			gmm.fit(r2)
			data = list()
			for gaussian in range(0,num_gaussians):
				temp = np.sqrt(gmm.covars_[gaussian]) * np.random.randn(gmm.weights_[gaussian]*1000,1) + gmm.means_[gaussian]
				data = data + np.concatenate(temp).tolist()

			(hist2, bins2) = np.histogram(data, bins=20, range=(min(data)-np.std(data),max(data)+np.std(data)))
			xx1 = np.linspace(min(data)-np.std(data), max(data)+np.std(data))
			
			pylab.hist(data, bins=20)
			#pylab.plot(bins1[0:len(bins1)-1],hist1, 'o', bins2[0:len(bins2)-1],hist2, 'x')
			#pylab.plot(bins1[0:len(bins1)-1],hist1, 'o', xx1, poly1(xx1),'-b' ,xx2, poly2(xx2), '-g', bins2[0:len(bins2)-1],hist2, 'x')
			pylab.show()
		self.rewards_this_iteration = list()
	def find_num_gaussians(self, poly1, poly1d, poly1dd):
		roots = poly1d.roots()
		it = 0
		num_gaussians = 0
		for r in roots:
			if poly1dd(r) < 0:
				if it >= len(roots)-1:
					break
				#check_gaussian_size()
				left_edge = roots[it-1]
				right_edge = roots[it+1]
				area = poly1.integral(left_edge, right_edge)
				if area > 1:
					num_gaussians += 1
			it += 1
		return num_gaussians

	def get_splines(self, rewards):
		(hist1, bins1) = np.histogram(rewards, bins=20, range=(min(rewards)-np.std(rewards),max(rewards)+np.std(rewards)))
		xx1 = np.linspace(min(rewards)-np.std(rewards), max(rewards)+np.std(rewards))
		spline = interpolate.InterpolatedUnivariateSpline(bins1[0:len(bins1)-1], hist1)
		splined = spline.derivative()
		splined = interpolate.InterpolatedUnivariateSpline(xx1, splined(xx1))
		splinedd = splined.derivative()
		splinedd = interpolate.InterpolatedUnivariateSpline(xx1, splinedd(xx1))
		return (spline, splined, splinedd)

	def add_gmm(self, gmm1, gmm2, sample_one=1, sample_two=1, add=False):
		if add == True:
			total_number = 0
			total = 0
			for gaussian in range(0, len(gmm2.means_)):	
				weight = gmm2.weights_[gaussian]
				mean = gmm2.means_[gaussian]
				total += weight*mean
				total_number += 1
			mean = mean/total_number

			for gaussian in range(0, len(gmm1.means_)):
				gmm1.means_[gaussian] += mean
			
			return gmm1

		data_gmm1 = list()
		data_gmm2 = list()
		for gaussian in range(0,len(gmm1.means_)):	
			temp = np.sqrt(gmm1.covars_[gaussian]) * np.random.randn(max(gmm1.weights_[gaussian]*10000*sample_one,1),1) + gmm1.means_[gaussian]
			data_gmm1 = data_gmm1 + np.concatenate(temp).tolist()
		for gaussian in range(0,len(gmm2.means_)):
			temp = np.sqrt(gmm2.covars_[gaussian]) * np.random.randn(max(gmm2.weights_[gaussian]*10000*sample_two,1),1) + gmm2.means_[gaussian]
			data_gmm2 = data_gmm2 + np.concatenate(temp).tolist()		
		#data_gmm1 = gmm1.sample(1000)
		#data_gmm2 = gmm2.sample(1000)
		if add == False:
			data_merged = data_gmm1 + data_gmm2
		#			data_merged = [x+y for x,y in zip(data_gmm1, data_gmm2)]
			

		(poly1, poly1d, poly1dd) = self.get_splines(data_merged)
		num_gaussians = self.find_num_gaussians(poly1, poly1d, poly1dd)
	
		gmm = mixture.gmm.GMM(n_components=num_gaussians)
		gmm.fit(data_merged)
		return gmm

	def subtract_gmm(self, gmm1, gmm2, sample_one=1, sample_two=1):
		data_gmm1 = list()
		data_gmm2 = list()
		for gaussian in range(0,len(gmm1.means_)):
			temp = np.sqrt(gmm1.covars_[gaussian]) * np.random.randn(max(gmm1.weights_[gaussian]*10000*sample_one,1),1) + gmm1.means_[gaussian]
			data_gmm1 = data_gmm1 + np.concatenate(temp).tolist()
		for gaussian in range(0,len(gmm2.means_)):
			temp = np.sqrt(gmm2.covars_[gaussian]) * np.random.randn(max(gmm2.weights_[gaussian]*10000*sample_two,1),1) + gmm2.means_[gaussian]
			data_gmm2 = data_gmm2 + np.concatenate(temp).tolist()
		#data_gmm1 = gmm1.sample(1000)
		#data_gmm2 = gmm2.sample(1000)
		#data_merged = [x-y for x,y in zip(data_gmm1, data_gmm2)]

		(probabilities, _) = gmm2.score_samples(data_gmm1)

		it = 0
		for p in probabilities:
			r = random.random()
			actual_p = math.exp(p)
			if actual_p > r:
				data_gmm1.pop(it)
				it -= 1
			it += 1
		#print data_gmm1
		data_merged = data_gmm1
		(poly1, poly1d, poly1dd) = self.get_splines(data_merged)
		num_gaussians = self.find_num_gaussians(poly1, poly1d, poly1dd)

		gmm = mixture.gmm.GMM(n_components=num_gaussians)
		gmm.fit(data_merged)
		return gmm	



	def add_state_actions_cheat(self, state, action):
		#Average time it takes to perfectly navigate to the goal.
		global navigation_time

		#Average time time it takes to perfectly perform the task.
		global task_time


		global nav_teleop_variance
		global nav_teleop_mean
		global nav_shared_variance
		global nav_shared_mean
		global nav_automa_variance
		global nav_automa_mean
		global nav_automa_failure
		global nav_automa_failure_scale
		global task_teleop_variance
		global task_teleop_mean
		global task_automa_variance
		global task_automa_mean
		global pack_arms_mean
		global pack_arms_variance
		global unpack_arms_mean
		global unpack_arms_variance

		if action == "NavTele":
			#self.QTable_additive_states[state][action].append({'V':0, 'mean':-nav_teleop_mean, 'std':nav_teleop_variance})
			self.QTable_additive_states[state][action].append({'V':0, 'mean':0, 'std':1})
			self.additive_states_probabilities[state][action].append(1.0)
			self.additive_states_probabilities_counts[state][action].append(1.0)
			if state not in self.rewards.keys():
				self.rewards[state] = dict()
			self.rewards[state][action] = list()
			self.rewards[state][action].append([])
		elif action == "NavShared":
			#self.QTable_additive_states[state][action].append({'V':0, 'mean':-nav_shared_mean, 'std':nav_shared_variance})
			self.QTable_additive_states[state][action].append({'V':0, 'mean':0, 'std':1})
			self.additive_states_probabilities[state][action].append(1.0)
			self.additive_states_probabilities_counts[state][action].append(1.0)
			if state not in self.rewards.keys():
				self.rewards[state] = dict()
			self.rewards[state][action] = list()
			self.rewards[state][action].append([])
		elif action == "NavAuto":
			#self.QTable_additive_states[state][action].append({'V':0, 'mean':-(nav_automa_mean), 'std':nav_automa_variance})
			self.QTable_additive_states[state][action].append({'V':0, 'mean':-(nav_automa_mean * nav_automa_failure_scale + pack_arms_mean), 'std':nav_automa_variance*nav_automa_failure_scale*nav_automa_failure_scale})
			#self.QTable_additive_states[state][action].append({'V':0, 'mean':0, 'std':1})
			self.additive_states_probabilities[state][action].append(.5)
			self.additive_states_probabilities_counts[state][action].append(1.0)
			if state not in self.rewards.keys():
				self.rewards[state] = dict()
			self.rewards[state][action] = list()
			self.rewards[state][action].append([])
			#self.additive_states_probabilities[state][action].append(.5)
			#self.additive_states_probabilities_counts[state][action].append(1.0)
			#self.rewards[state][action].append([])
		elif action == "SwitchAuto":
			#self.QTable_additive_states[state][action].append({'V':0, 'mean':-task_automa_mean, 'std':task_automa_variance})
			self.QTable_additive_states[state][action].append({'V':0, 'mean':0, 'std':1})
			self.additive_states_probabilities[state][action].append(1.0)
			self.additive_states_probabilities_counts[state][action].append(1.0)
			if state not in self.rewards.keys():
				self.rewards[state] = dict()
			self.rewards[state][action] = list()
			self.rewards[state][action].append([])
		elif action == "SwitchTele":
			#self.QTable_additive_states[state][action].append({'V':0, 'mean':-task_teleop_mean, 'std':task_teleop_variance})
			self.QTable_additive_states[state][action].append({'V':0, 'mean':0, 'std':1})
			self.additive_states_probabilities[state][action].append(1.0)
			self.additive_states_probabilities_counts[state][action].append(1.0)
			if state not in self.rewards.keys():
				self.rewards[state] = dict()
			self.rewards[state][action] = list()
			self.rewards[state][action].append([])
		elif action == "PackArms":
			#self.QTable_additive_states[state][action].append({'V':0, 'mean':-pack_arms_mean, 'std':pack_arms_variance})
			self.QTable_additive_states[state][action].append({'V':0, 'mean':0, 'std':1})
			self.additive_states_probabilities[state][action].append(1.0)
			self.additive_states_probabilities_counts[state][action].append(1.0)
			if state not in self.rewards.keys():
				self.rewards[state] = dict()
			self.rewards[state][action] = list()
			self.rewards[state][action].append([])
		elif action == "UnpackArms":
			#self.QTable_additive_states[state][action].append({'V':0, 'mean':-unpack_arms_mean, 'std':unpack_arms_variance})
			self.QTable_additive_states[state][action].append({'V':0, 'mean':0, 'std':1})
			self.additive_states_probabilities[state][action].append(1.0)
			self.additive_states_probabilities_counts[state][action].append(1.0)
			if state not in self.rewards.keys():
				self.rewards[state] = dict()
			self.rewards[state][action] = list()
			self.rewards[state][action].append([])
		else:
			#self.QTable_additive_states[state][action].append({'V':0, 'mean':0.01, 'std':0.01})
			self.QTable_additive_states[state][action].append({'V':0, 'mean':0, 'std':1})
			self.additive_states_probabilities[state][action].append(1.0)
			self.additive_states_probabilities_counts[state][action].append(1.0)
			if state not in self.rewards.keys():
				self.rewards[state] = dict()
			self.rewards[state][action] = list()
			self.rewards[state][action].append([])


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
