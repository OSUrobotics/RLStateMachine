#!/usr/bin/env python

import rospy
import threading
import traceback
from contextlib import contextmanager
import random
import smach

__all__ = ['StateMachineRL']

### State Machine class
class StateMachineRL(smach.container.Container):
    """StateMachine
    
    This is a finite state machine smach container. Note that though this is
    a state machine, it also implements the L{smach.State}
    interface, so these can be composed hierarchically, if such a pattern is
    desired.

    States are added to the state machine as 3-tuple specifications:
     - label
     - state instance
     - transitions

    The label is a string, the state instance is any class that implements the
    L{smach.State} interface, and transitions is a dictionary mapping strings onto
    strings which represent the transitions out of this new state. Transitions
    can take one of three forms:
     - OUTCOME -> STATE_LABEL
     - OUTCOME -> None (or unspecified)
     - OUTCOME -> SM_OUTCOME
    """
    def __init__(self,
            outcomes,
            default_outcome,
            input_keys = [],
            output_keys = [],
            outcome_map = {},
            outcome_cb = None,
            child_termination_cb = None,
            ):
        """Constructor for smach Concurrent Split.

        @type outcomes: list of strings
        @param outcomes: The potential outcomes of this state machine.

        @type default_outcome: string
        @param default_outcome: The outcome of this state if no elements in the 
        outcome map are satisfied by the outcomes of the contained states.


        @type outcome_map: list
        @param outcome_map: This is an outcome map for determining the
        outcome of this container. Each outcome of the container is mapped
        to a dictionary mapping child labels onto outcomes. If none of the
        child-outcome maps is satisfied, the concurrence will terminate
        with thhe default outcome.
        
        For example, if the and_outcome_map is:
            {'succeeded' : {'FOO':'succeeded', 'BAR':'done'},
             'aborted' : {'FOO':'aborted'}}
        Then the concurrence will terimate with outcome 'succeeded' only if
        BOTH states 'FOO' and 'BAR' have terminated
        with outcomes 'succeeded' and 'done', respectively. The outcome
        'aborted' will be returned by the concurrence if the state 'FOO'
        returns the outcome 'aborted'. 

        If the outcome of a state is not specified, it will be treated as
        irrelevant to the outcome of the concurrence

        If the criteria for one outcome is the subset of another outcome,
        the container will choose the outcome which has more child outcome
        criteria satisfied. If both container outcomes have the same
        number of satisfied criteria, the behavior is undefined.

        If a more complex outcome policy is required, see the user can
        provide an outcome callback. See outcome_cb, below.

        @type child_termination_cb: callale
        @param child_termination_cb: This callback gives the user the ability
        to force the concurrence to preempt running states given the
        termination of some other set of states. This is useful when using
        a concurrence as a monitor container. 

        This callback is called each time a child state terminates. It is
        passed a single argument, a dictionary mapping child state labels
        onto their outcomes. If a state has not yet terminated, it's dict
        value will be None.

        This function can return three things:
         - False: continue blocking on the termination of all other states
         - True: Preempt all other states
         - list of state labels: Preempt only the specified states

        I{If you just want the first termination to cause the other children
        to terminate, the callback (lamda so: True) will always return True.}

        @type outcome_cb: callable
        @param outcome_cb: If the outcome policy needs to be more complicated
        than just a conjunction of state outcomes, the user can supply
        a callback for specifying the outcome of the container.

        This callback is called only once all child states have terminated,
        and it is passed the dictionary mapping state labels onto their
        respective outcomes.

        If the callback returns a string, it will treated as the outcome of
        the container.

        If the callback returns None, the concurrence will first check the
        outcome_map, and if no outcome in the outcome_map is satisfied, it
        will return the default outcome.

        B{NOTE: This callback should be a function ONLY of the outcomes of
        the child states. It should not access any other resources.} 

        """
        smach.container.Container.__init__(self, outcomes, input_keys, output_keys)

        # List of concurrent states
        self._states = {}
        self._threads = {}
        self._remappings = {}

        if not (default_outcome or outcome_map or outcome_cb):
            raise smach.InvalidStateError("Concurrence requires an outcome policy")

        # Initialize error string
        errors = ""

        # Check if default outcome is necessary
        if default_outcome != str(default_outcome):
            errors += "\n\tDefault outcome '%s' does not appear to be a string." % str(default_outcome)
        if default_outcome not in outcomes:
            errors += "\n\tDefault outcome '%s' is unregistered." % str(default_outcome)

        # Check if outcome maps only contain outcomes that are registered
        for o in outcome_map:
            if o not in outcomes:
                errors += "\n\tUnregistered outcome '%s' in and_outcome_map." % str(o)

        # Check if outcome cb is callable
        if outcome_cb and not hasattr(outcome_cb,'__call__'):
            errors += "\n\tOutcome callback '%s' is not callable." % str(outcome_cb)

        # Check if child termination cb is callable
        if child_termination_cb and not hasattr(child_termination_cb,'__call__'):
            errors += "\n\tChild termination callback '%s' is not callable." % str(child_termination_cb)

        # Report errors
        if len(errors) > 0:
            raise smach.InvalidStateError("Errors specifying outcome policy of concurrence: %s" % errors)

        # Store outcome policies
        self._default_outcome = default_outcome
        self._outcome_map = outcome_map
        self._outcome_cb = outcome_cb
        self._child_termination_cb = child_termination_cb
        self._child_outcomes = {}
        self._labels = []

        # Condition variables for threading synchronization
        self._user_code_exception = False
        self._done_cond = threading.Condition()

    ### Construction methods
    @staticmethod
    def add(label, state, remapping={}):
        """Add state to the opened concurrence.
        This state will need to terminate before the concurrence terminates.
        """
        # Get currently opened container
        self = StateMachineRL._currently_opened_container()

        # Store state
        self._states[label] = state
        self._remappings[label] = remapping
        self._labels.append(label)
        return state

    ### Internals
    def _set_current_state(self, state_label):
        if state_label is not None:
            # Store the current label and states 
            self._current_label = state_label
            self._current_state = self._states[state_label]
            self._current_outcome = None
        else:
            # Store the current label and states 
            self._current_label = None
            self._current_state = None
            self._current_outcome = None

    ### State Interface
    def execute(self, parent_ud = smach.UserData()):
        """Run the state machine on entry to this state.
        This will set the "closed" flag and spin up the execute thread. Once
        this flag has been set, it will prevent more states from being added to
        the state machine. 
        """


        # Initialize preempt state
        self._preempted_label = None
        self._preempted_state = None

        # Set initial state. TODO:Do this using RL.
        val = random.randrange(0,len(self._states),1)
        label = self._labels[val]
        self._set_current_state(label)

        # Copy input keys
        self._copy_input_keys(parent_ud, self.userdata)

        execution_outcome = self._current_state.execute((smach.Remapper(
                self.userdata,
                self._states[label].get_registered_input_keys(),
                self._states[label].get_registered_output_keys(),
                self._remappings[label])))


        # Spew some info
        smach.loginfo("RL Stuff '%s' with userdata: \n\t%s" %
                (self._current_label, list(self.userdata.keys())))

        # Copy output keys
        self._copy_output_keys(self.userdata, parent_ud)

        # We're no longer running
        self._is_running = False

        for (container_outcome, outcomes) in ((k,self._outcome_map[k]) for k in self._outcome_map):
        		if execution_outcome in outcomes:
        			 break
        
        return container_outcome

    ## Preemption management
    def request_preempt(self):
        """Propagate preempt to currently active state.
        
        This will attempt to preempt the currently active state.
        """
        with self._state_transitioning_lock:
            # Aleways Set this container's preempted flag
            self._preempt_requested = True
            # Only propagate preempt if the current state is defined
            if self._current_state is not None:
                self._preempt_current_state()

    def _preempt_current_state(self):
        """Preempt the current state (might not be executing yet).
        This also resets the preempt flag on a state that had previously received the preempt, but not serviced it."""
        if self._preempted_state != self._current_state:
            if self._preempted_state is not None:
                # Reset the previously preempted state (that has now terminated)
                self._preempted_state.recall_preempt()

            # Store the label of the currently active state
            self._preempted_state = self._current_state
            self._preempted_label = self._current_label

            # Request the currently active state to preempt
            try:
                self._preempted_state.request_preempt()
            except:
                smach.logerr("Failed to preempt contained state '%s': %s" % (self._preempted_label, traceback.format_exc()))

    ### Container interface
    def get_children(self):
        return self._states

    def __getitem__(self,key):
        if key not in self._states:
            smach.logerr("Attempting to get state '%s' from StateMachine container. The only available states are: %s" % (key, str(list(self._states.keys()))))
            raise KeyError()
        return self._states[key]

    def get_active_states(self):
        return [label for (label,outcome) in ((k,self._child_outcomes[k]) for k in self._child_outcomes) if outcome is None]

    def get_initial_states(self):
        return list(self._states.keys())

    def get_internal_edges(self):
        int_edges = []
        for (container_outcome, outcomes) in ((k,self._outcome_map[k]) for k in self._outcome_map):
            for labels in self._labels:
	            int_edges.append((outcomes, labels, container_outcome))
        return int_edges

    ### Validation methods
    def check_state_spec(self, label, state, transitions):
        """Validate full state specification (label, state, and transitions).
        This checks to make sure the required variables are in the state spec,
        as well as verifies that all outcomes referenced in the transitions
        are registered as valid outcomes in the state object. If a state
        specification fails validation, a L{smach.InvalidStateError} is
        thrown.
        """
        # Make sure all transitions are from registered outcomes of this state
        registered_outcomes = state.get_registered_outcomes()
        for outcome in transitions:
            if outcome not in registered_outcomes:
                raise smach.InvalidTransitionError("Specified outcome '"+outcome+"' on state '"+label+"', which only has available registered outcomes: "+str(registered_outcomes))

    def check_consistency(self):
        """Check the entire state machine for consistency.
        This asserts that all transition targets are states that are in the
        state machine. If this fails, it raises an L{InvalidTransitionError}
        with relevant information.
        """
        # Construct a set of available states

    ### Introspection methods
    def is_running(self):
        """Returns true if the state machine is running."""
        return self._is_running
