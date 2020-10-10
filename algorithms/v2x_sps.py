import numpy as np
import random

class SemiPersistentScheduling:
    """
    Semi Persistent Scheduling(SPS) implementation, this class receives the observation and user id for each action
    """
    def __init__(self, user, selection_window, RSSI_threshold):
        self.user_id = user
        self.selection_window_size = selection_window  #  Defines the interval T2-T1 in subframes not in ms.
        self.RSSI_threshold = RSSI_threshold
        # txSubframe depicts the action will be taken by the agent.
        self.txSubframe = random.randint(0, self.selection_window_size)  # T1 will be added in the simulator.
        # Reselection counter is initialized at first.
        self.reselection_counter = random.randint(5, 15)

        # Previous action value.
        self.prev_action = self.txSubframe
        self.inc_dB = 3 # increase the threshold value 3 dB
        # UE sense ~-117 dB and if there is a transmission at the resource it sense ~ -200dB

        self.prob_resource_keep = 0.8

    def choose_new_resource(self, selection_window):
        """
        Agent receives an averaged RSSI values from the simulator every time. Then, it just runs
        the algorithm.
        :param sensing_window: averaged RSSI values, UE decides from this interval.
        :return: subframe index
        """

        # IF a subframe is higher than the threshold, add it to the list.
        # This should be higher than the %20 of the resources available in the selection window.
        # If a UE could not find a resources then, it just decrease the threshold.
        #
        sB = []
        tmp_threshold = self.RSSI_threshold
        # Minimum number of available resources should be higher than %20 of the selection window.
        min_sA = len(selection_window)/5
        sA = {}
        while len(sA) < min_sA:
            sA = {}
            for subframe in range(len(selection_window)):
                if self.prev_action == subframe:
                    continue
                if selection_window[subframe] < tmp_threshold:
                    sA[subframe] = selection_window[subframe]
                   # sA.append(subframe, selection_window[subframe])

            tmp_threshold += self.inc_dB

        # Sorted sA includes the available resources in descending orders
        sorted_sA = sorted(sA.items(), key=lambda x: x[1])
        min_len = min(min_sA, len(sA))
        for k, v in sorted_sA:
            sB.append(k)
            if len(sB) >= min_len:
                break
        action = random.choice(sB)


#        sorted(sA.items(), key =
 #            lambda kv:(kv[1], kv[0]))
        #sorted_sA = sA.sort(reverse=True)
        #num_best_resource = min(min_sA, len(sA))
        #rand = random.randint(0, len(selection_window))
        #index = rand % num_best_resource
        #subframe_index = sorted_sA[index]


        # Update the reselection counter value
        # self.reselection_counter = random.randint(5, 15)

        return action

    def step(self, selection_window):
        """
        Receives the averaged RSSI values from the simulator and decide for an action.
        Sps algorithm.
        :param selection_window:
        :return: action(subframe to be exploited to send a message.)
        """
        action = None
        # return self.user_id+1

        if self.reselection_counter != 0:
            # Use the previous selected subframe index
            # And reduce the reselection counter.
            action = self.prev_action
            self.reselection_counter -= 1
        else:
            self.reselection_counter = random.randint(5, 16)

            if random.random() < self.prob_resource_keep:
                # Then use the same subframe for the transmission.
                action = self.prev_action
            else:
                action = self.choose_new_resource(selection_window)
                self.prev_action = action

        if action is None:
            print("Olmaz oyle sacma sey")

        return action
