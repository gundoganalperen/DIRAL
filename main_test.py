from __future__ import division
import time
import sys
import yaml
import numpy as np
import os
from algorithms.drl_drqn import DRQN
from utils.memory import Memory
from collections import defaultdict, deque

from envs.test_env import TestEnv
from utils.misc import calculate_ia_penalty

def marl_test(config):

    experiment_name = config.setdefault("experiment_name", "")
    time_slots = config.setdefault("time_slots", 10000)
    simulations = config.setdefault("simulations", 3)

    memory_size = config.setdefault("memory_size", 1200)
    pretrain_length = config.setdefault("pretrain_length", 6)
    step_size = config.setdefault("step_size", 5)
    save_freq = config.setdefault("save_freq", 1000)
    save_results = config.setdefault("save_results", True)
    save_model = config.setdefault("save_model", False)
    load_model = config.setdefault("load_model", False)
    load_slot = config.setdefault("load_slot", 4999)
    training = config.setdefault("training", False)
    episode_interval = config.setdefault("episode_interval", 25)
    explore_step = config.setdefault("explore", 2000)
    greedy_step = config.setdefault("greedy", 20000)
    training_stop = config.setdefault("training_stop", 20000)  # Stop the training after these time step.
    train_after_episode = config.setdefault("train_after_episode", False)  # Train after each episode in stead of training after each time slot.
    global_reward_avg = config.setdefault("global_reward_avg", False)  # Train after each episode in stead of training after each time slot.
    save_positions = config.setdefault("save_positions", False)  # Train after each episode in stead of training after each time slot.
    enable_channel = config.setdefault("enable_channel", False)  # Train after each episode in stead of training after each time slot.

    batch_size = config["RLAgent"]["batch_size"]
    ia_penalty_enable = config.setdefault("ia_penalty_enable", False)
    ia_averaging = config.setdefault("ia_averaging", False)


    for simulation in range(simulations):
        print("-=-=-=-=-=-=-=-=-=-=-= experiment_name: " + experiment_name + " SIMULATION " + str(simulation + 1) + " =-=-=-=-=-=-=-=-=-=-=-")
        # Initialize the env.
        env = TestEnv(**config["EnvironmentTest"])

        if ia_penalty_enable:
            ia_penalty_threshold = config.setdefault("ia_penalty_threshold", 5)
            ia_penalty_value = config.setdefault("ia_penalty_value", -10)
            ia_penalty_counter = {}
            previous_actions = {}  # store the previous taken action by the UE.
            num_users = env.get_total_users()
            for user in range(num_users):
                ia_penalty_counter[user] = 0
                previous_actions[user] = -1

            # Initialize the agen

        mainDRQN = DRQN(env, name=experiment_name, total_episodes=time_slots/episode_interval, **config["RLAgent"])
        #mainDRQN = DeepRecurrentQNetwork(env=env, name=experiment_name, **config["RLAgent"])
        if load_model:
            print("Load model DRQN time step " + str(load_slot))
            save_dir = "save_model/" + "test/"
            mainDRQN.load_model(save_dir, load_slot)

        # this is experience replay buffer(deque) from which each batch will be sampled and fed to the neural network for training
        memory = Memory(max_size=memory_size)

        log_reward_slot = []
        log_actions_slot = []
        log_ia_slot = []
        sum_ia_prev = 0

        log_x_positions = []
        start_time = time.time()
        episode = 0  # Used to update the greediness of the algorithm
        # cumulative reward
        cum_r = [0]
        cum_r_slots = [0]

        # cumulative collision
        cum_collision = [0]
        cum_collision_slots = [0]
        # this is our input buffer which will be used for  predicting next Q-values
        history_input = deque(maxlen=step_size)
       # env.network.reset_ia()
        # to sample random actions for each user
        action = env.sample()

        #obs = env.step(action)
        obs, rews = env.my_step(action, 0)
        rews = list(rews)
        state = env.obtain_state(obs, action, rews)
        # reward = [i[1] for i in obs[:num_users]]
        num_users = env.get_total_users()
        num_channels = env.get_action_space()
        ##############################################
        for ii in range(pretrain_length*step_size*5):
            action = env.sample()
            if enable_channel:
                obs, reward = env.my_step_ch(action,
                                             0)  # obs is a list of tuple with [(ACK,REW) for each user ,(CHANNEL_RESIDUAL_CAPACITY_VECTOR)]
            else:
                #obs, reward = env.my_step(
                #    action, 0)  # obs is a list of tuple with [(ACK,REW) for each user ,(CHANNEL_RESIDUAL_CAPACITY_VECTOR)]
                obs, reward = env.my_step_design(action, 0)

            # obs is a list of tuple with [[(ACK,REW) for each user] ,CHANNEL_RESIDUAL_CAPACITY_VECTOR]
            next_state = env.obtain_state(obs, action, rews)
            #next_state = env.state_generator(action, obs)
            memory.add((state, action, rews, next_state))
            state = next_state
            history_input.append(state)

            ##############################################
        # TODO: now load the positions
        env.load_saved_positions()
        for time_step in range(time_slots):
            #initializing action vector
            action = np.zeros([num_users], dtype=np.int32)

            #converting input historskyy into numpy array
            # TODO: enable below for lstm
            state_vector = np.array(history_input)  #  LSTM
            #  state_vector = state  #  DQN
            for each_user in range(num_users):
                #action[each_user] = mainDRQN.infer_action(each_user, state_vector=state_vector, time_slot=time_step)
                if time_step < explore_step and not load_model: # and 0:
                    action[each_user] = mainDRQN.infer_action(each_user, state_vector=state_vector, episode=episode,
                                                              policy="explore")

                elif time_step < greedy_step and not load_model: # and 0:
                    action[each_user] = mainDRQN.infer_action(each_user, state_vector=state_vector, episode=episode)
                else:
                    action[each_user] = mainDRQN.infer_action(each_user, state_vector=state_vector, episode=episode, policy="greedy")

            # taking action as predicted from the q values and receiving the observation from the envionment
            # obs = env.step(action)           # obs is a list of tuple with [(ACK,REW) for each user ,(CHANNEL_RESIDUAL_CAPACITY_VECTOR)]
            if save_positions:
                user_pos = env.get_x_pos()
                log_x_positions.append(user_pos)
            if enable_channel:
                obs, reward = env.my_step_ch(action, time_step)           # obs is a list of tuple with [(ACK,REW) for each user ,(CHANNEL_RESIDUAL_CAPACITY_VECTOR)]
            else:
                obs, reward = env.my_step(action, time_step)           # obs is a list of tuple with [(ACK,REW) for each user ,(CHANNEL_RESIDUAL_CAPACITY_VECTOR)]
                #obs, reward = env.my_step_design(action, time_step)
                # TODO: update the env topology after each step.
            log_actions_slot.append(action)
            ia = env.network.get_information_age(time_step)
            ia_sum = calculate_ia_penalty(ia)
            log_ia_slot.append(ia)
            if ia_averaging:  # ia based penalty to the reward
                ia_penalty = 0
                if ia_sum > sum_ia_prev:
                    ia_penalty = -1
                elif ia_sum < sum_ia_prev:
                    ia_penalty = 1

                sum_ia_prev = ia_sum

            # Generate next state from action and observation
            # next_state = env.state_generator(action, obs)  used for DQN
            next_state = env.obtain_state(obs, action, reward, episode, mainDRQN.get_eps())
            #	print (next_state)

            # reward for all users given by environment
            #reward = [i[1] for i in obs[:num_users]]

            # calculating sum of rewards
            sum_r = np.sum(reward)

            #calculating cumulative reward
            cum_r.append(cum_r[-1] + sum_r)
            cum_r_slots.append(cum_r_slots[-1] + sum_r)

            #If NUM_CHANNELS = 2 , total possible reward = 2 , therefore collision = (2 - sum_r) or (NUM_CHANNELS - sum_r)
            collision = num_channels - sum_r

            #calculating cumulative collision
            cum_collision.append(cum_collision[-1] + collision)
            cum_collision_slots.append(cum_collision_slots[-1] + collision)
            #############################
            #  for co-operative policy we will give reward-sum to each user who have contributed
            #  to play co-operatively and rest 0
            # NOTE: I think, I do not need that part since I already use positive and negative reward.

            for i in range(len(reward)):  # for each user we have this.
                #if reward[i] > 0:
                if ia_averaging:
                    # add penalty based on the direction of the Information age.
                    reward[i] += ia_penalty

                if ia_penalty_enable:
                    if reward[i] < 1 and action[i] == previous_actions[i]:
                        ia_penalty_counter[i] += 1
                    else:
                        ia_penalty_counter[i] = 0

                    if ia_penalty_counter[i] > ia_penalty_threshold:
                        reward[i] = ia_penalty_value

                    previous_actions[i] = action[i]

                if global_reward_avg:
                    reward[i] = reward[i] + sum_r/len(reward)  # Add the average total reward to each UE.

            #############################
            #reward = reward*2  # Add the average total reward to each UE.
            log_reward_slot.append(sum_r)
            #	print (reward)
            #	print("EPOCH " + str(time_step))

            # add new experiences into the memory buffer as (state, action , reward , next_state) for training
            memory.add((state, action, reward, next_state))

            state = next_state
            #add new experience to generate input-history sequence for next state
            history_input.append(state)

            #  Start training.
            if not train_after_episode:
                if time_step < training_stop and training: #and not load_model:
                    mainDRQN.train(memory, time_step)

            if time_step%(episode_interval) == episode_interval-1:
                print("Time step " + str(time_step) + " epsilon " + str(mainDRQN.get_eps())
                      + " cum Collison " + str(cum_collision[episode_interval]) + " sum reward " + str(cum_r[episode_interval]) + " total time " + str(time.time()-start_time) )
                cum_r = [0]
                cum_collision = [0]
                episode += 1
                # Updates the velocity of the vehicles if activated
                env.update_velocity()
               # ia = env.network.get_information_age(time_step)
                if train_after_episode and time_step > (batch_size+10) and training:
                    mainDRQN.train(memory, time_step)

            if time_step%save_freq == save_freq-1:
                # Save the collisions
                if save_results:
                    print("save results for timestep ", time_step + 1)
                    save_dir = "save_results/" + "test/"
                    save_dir = save_dir + experiment_name
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                   # filename = save_dir + "/collisions" + "_" + str(time_step) +"_sim"+str(simulation)
                   # np.save(filename, np.asarray(cum_collision_slots))
                    filename = save_dir + "/rewards" + "_sim"+str(simulation)
                    np.save(filename, np.asarray(log_reward_slot))
                    filename = save_dir + "/actions" + "_sim"+str(simulation)
                    np.save(filename, np.asarray(log_actions_slot))
                  #  filename = save_dir + "/time_step" + "_" + str(time_step)+"_sim"+str(simulation)
                  #  np.save(filename, np.asarray(str(time.time()-start_time)))
                    filename = save_dir + "/positions" + "_sim"+str(simulation)
                    np.save(filename, np.asarray(log_x_positions))
                    #filename = save_dir + "/ia" + "_sim"+str(simulation)
                    #np.save(filename, np.asarray(log_ia_slot))
                    #"_" + str(time_step)+

                if save_model:
                    print("save model for timestep ", time_step + 1)
                    save_dir = "save_model/" + "test/"
                    #save_dir = save_dir
                    mainDRQN.save_model(save_dir, time_step,simulation)


if __name__ == '__main__':
    # NOTE: This part should be commented to be able to debug in Pycharm.
    #if len(sys.argv) < 2:
    #    print("Run: python <script> <config>")
    #sys.exit(1)
    #script = sys.argv[0]
    #try:
    #   config = yaml.load(open(sys.argv[1]))
    #except:
    #   config = {}

    #config = yaml.load(open("configs/test/drqn/5ue_4r_softmax.yaml"))
    experiments = []


    ##  Test 2 check discount factor impact  ###
    experiments.append("configs/4ue_3r_toy/config_toy_4ue_3r_tests_db_r2_b20_mg_o_index_dis_03.yaml")


    # # =======
    for i in range(len(experiments)):
        config = yaml.load(open(experiments[i]))

        experiment_name = config.setdefault("experiment_name", "")
        realness = config.setdefault("realness", False)
        if realness:
            print("This should never happen!!!!")
        else:
            marl_test(config)

