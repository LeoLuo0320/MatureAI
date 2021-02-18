try:
    from malmo import MalmoPython
except:
    import MalmoPython
    import malmoutils

from numpy.random import randint
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import time
import json
import math
import pathlib
import gym, ray
from MapGenerator import OBS_SIZE, MAX_EPISODE_STEPS, Map
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo

DIAMOND_POS = []
DESTINATION_Z = 10000

class MinecraftRunner(gym.Env):

    def __init__(self, env_config):
        # Static Parameters
        self.obs_size = OBS_SIZE
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.log_frequency = 5
        self.action_dict = {
            0: 'move 1',  # Move one block forward
            1: 'turn 1',  # Turn 90 degrees to the right  # Turn 90 degrees to the left
            2: 'turn -1',
            3: 'use 1'
        }

        # Rllib Parameters
        self.action_space = Discrete(len(self.action_dict))
        self.observation_space = Box(0, 1, shape=(2 * self.obs_size * self.obs_size,), dtype=np.float32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        self.obs = None
        self.open_gate = False
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []
        self.shortest_to_dest = DESTINATION_Z

    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        # Reset Malmo
        world_state = self.init_malmo()

        # Reset Variables
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0

        # Log
        if len(self.returns) > self.log_frequency + 1 and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get Observation
        self.obs, self.open_gate = self.get_observation(world_state)

        self.shortest_to_dest = DESTINATION_Z

        return self.obs

    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.GetXML(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))  # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission(my_mission, my_clients, my_mission_record, 0, 'MatureRunner')
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)

        return world_state

    def obs_diamond(self, agent_x, agent_z):
        # Get observation matrix and agent row/col
        sight = int((OBS_SIZE - 1) / 2)
        diamond_obs = np.zeros((OBS_SIZE,OBS_SIZE))
        agent_row = int(OBS_SIZE/2)
        agent_col = int(OBS_SIZE/2)

        # Mark diamond position 1
        for diamond_x, diamond_z in DIAMOND_POS:
            x_diff = diamond_x - int(agent_x)
            z_diff = diamond_z - int(agent_z)
            check_x = -sight <= x_diff <= sight
            check_z = -sight <= z_diff <= sight
            if check_x and check_z:
                diamond_row = z_diff + agent_row
                diamond_col = x_diff + agent_col
                diamond_obs[diamond_row, diamond_col] = 1

        return diamond_obs

    def update_diamond_list(self, agent_x, agent_z):
        flag = False
        for diamond_x, diamond_z in DIAMOND_POS:
            if diamond_x == agent_x and diamond_z == agent_z:
                DIAMOND_POS.remove((diamond_x, diamond_z))
                flag = True
        return flag

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <int> index of the action to take


        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """

        # Get Action
        command = self.action_dict[action]
        if command != 'attack 1' or self.open_gate:
            self.agent_host.sendCommand(command)
            time.sleep(.1)
            self.episode_step += 1

        # Get Observation
        old_shortest = self.shortest_to_dest # Used for giving reward of moving to the destination
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs, self.open_gate = self.get_observation(world_state)

        # Get Done
        done = not world_state.is_mission_running

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            # print("r", r)
            # print("value: ", r.getValue())
            # input("Enter: ")
            reward += r.getValue()

        # Reward of moving towards to the destination
        if old_shortest > self.shortest_to_dest:
            reward += 1
        elif old_shortest < self.shortest_to_dest:
            reward -= 1

        self.episode_return += reward

        return self.obs, reward, done, dict()

    def get_observation(self, world_state):
        """
        Use the agent observation API to get a flattened 2 x 5 x 5 grid around the agent.
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array> the state observation
            allow_break_action: <bool> whether the agent is facing a diamond
        """
        obs = np.zeros((2 * self.obs_size * self.obs_size, ))
        open_gate = False

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)
                while 'floorAll' not in observations:
                    time.sleep(.1)  # maybe increment this each time it fails
                    world_state = self.agent_host.getWorldState()
                    msg = world_state.observations[-1].text
                    observations = json.loads(msg)

                # Get observation
                # Get block type
                grid = observations['floorAll']
                # print("grid: ", grid)
                # input("Enter: ")

                # Get agent position
                agent_x = observations['XPos']
                agent_z = observations['ZPos']

                # Update shortest distance to destination if current distance is shorter
                if DESTINATION_Z-agent_z < self.shortest_to_dest:
                    self.shortest_to_dest = DESTINATION_Z-agent_z

                for i, x in enumerate(grid):
                    obs[i] = x == 'jungle_fence_gate'

                # Get diamond observation and update diamond position
                obs = obs.reshape((2, self.obs_size, self.obs_size))
                obs[0] = self.obs_diamond(agent_x, agent_z)  # Get diamond observation
                obs = obs.reshape((2, self.obs_size, self.obs_size))

                # Remove collected diamond's position from the list to avoid repeat reward
                self.update_diamond_list(agent_x, agent_z)

                # Rotate observation with orientation of agent
                yaw = observations['Yaw']

                if yaw >= 225 and yaw < 315:
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif yaw >= 315 or yaw < 45:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif yaw >= 45 and yaw < 135:
                    obs = np.rot90(obs, k=3, axes=(1, 2))

                obs = obs.flatten()
                if 'LineOfSight' in observations.keys():
                    open_gate = observations['LineOfSight']['type'] == "jungle_fence_gate"
            break

        return obs, open_gate

    def GetXML(self):
        global DIAMOND_POS
        XMLmap, DIAMOND_POS = Map()
        return XMLmap

    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns[1:], box, mode='same')
        plt.clf()
        plt.plot(self.steps[1:], returns_smooth)
        plt.title('Mature Runner')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps[1:], self.returns[1:]):
                f.write("{}\t{}\n".format(step, value))


if __name__ == '__main__':

    ray.init()
    trainer = ppo.PPOTrainer(env=MinecraftRunner, config={
        'env_config': {},  # No environment parameters to configure
        'framework': 'torch',  # Use pyotrch instead of tensorflow
        'num_gpus': 0,  # We aren't using GPUs
        'num_workers': 0  # We aren't using parallelism
    })

    answer = input("Use last training result[Y/N]?")
    if answer.lower() == "y":
        while True:
            dir_path = input("Training data path:")
            if os.path.exists(dir_path):
                trainer.load_checkpoint(dir_path)
                break
            else:
                print(f"Invalid path:{dir_path}")

    current_directory = pathlib.Path(__file__).parent.absolute()

    i = 0
    while True:
        result = trainer.train()
        print(result)
        i += 1
        if i % 1 == 0:
            checkpoint = trainer.save_checkpoint(current_directory)
            print("checkpoint saved at", checkpoint)
