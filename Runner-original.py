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
import random
import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo


class MinecraftRunner(gym.Env):

    def __init__(self, env_config):
        # Static Parameters
        # self.size = 50
        # self.reward_density = .1
        # self.penalty_density = .02
        self.obs_size = 5
        self.max_episode_steps = 100
        self.log_frequency = 10
        self.action_dict = {
            0: 'move 1',  # Move one block forward
            1: 'use 1',
            2: 'turn 1',  # Turn 90 degrees to the right
            3: 'turn -1'  # Turn 90 degrees to the left
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
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

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
        if command != 'use 1' or self.open_gate:
            self.agent_host.sendCommand(command)
            time.sleep(.2)
            if command == 'use 1':
                self.agent_host.sendCommand('move 1')
        self.episode_step += 1

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs, self.open_gate = self.get_observation(world_state)

        # Get Done
        done = not world_state.is_mission_running

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
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

                # Get observation
                grid = observations['floorAll']
                # print("grid: ", grid)
                # input("Enter: ")
                for i, x in enumerate(grid):
                    obs[i] = x == 'jungle_fence_gate' or x == 'stone'

                # Rotate observation with orientation of agent
                obs = obs.reshape((2, self.obs_size, self.obs_size))
                yaw = observations['Yaw']

                if yaw >= 225 and yaw < 315:
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif yaw >= 315 or yaw < 45:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif yaw >= 45 and yaw < 135:
                    obs = np.rot90(obs, k=3, axes=(1, 2))

                obs = obs.flatten()
                open_gate = 'jungle_fence_gate' in grid

            break

        return obs, open_gate

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

    # def GetXML(self):
    #     TNT_and_TRIGGERS = ""
    #     trigger_z = [i for i in range(-48, 50, 7)]
    #
    #     # Stage Reward
    #     checkpoint_loc = [[2, -47], [2, -45], [2, -40], [2, -35],[2, -30], [2, -25], [2, -20], [2, -15], [2, -10], [2, -5]]
    #
    #     for x,z in checkpoint_loc:
    #         checkpoint_setup = f"<DrawBlock x2='2' y1='49' y2='49' z1='51' z2='51' type='diamond_ore' />"
    #
    #     destination = f"<DrawBlock x='0' y='49' z='-40' type='emerald_block' />"
    #
    #     # Obstacles
    #     # pillar_setup = ""
    #     # pillar_loc = []
    #     # num_of_pillar = 5
    #     # for i in range(num_of_pillar):
    #     #     xloc = random.randint(-2, 2)
    #     #     zloc = random.randint(-48, 50)
    #     #     loc = [xloc, zloc]
    #     #     while loc in pillar_loc:
    #     #         xloc = random.randint(-2, 2)
    #     #         zloc = random.randint(-45, 50)
    #     #         loc = [xloc, zloc]
    #     #     pillar_loc.append(loc)
    #     #     pillar_setup += f"<DrawBlock x='{xloc}' y='50' z='{zloc}' type='sandstone'/> \n"
    #
    #     fence_setup = ""
    #     fence_loc = []
    #
    #     zloc_list = [i for i in range(-48, -40, 2)]
    #     for zloc in zloc_list:
    #         xloc = random.randint(-1, 1)
    #         loc = [xloc, zloc]
    #         while loc in fence_loc:
    #             xloc = random.randint(-2, 3)
    #             loc = [xloc, zloc]
    #         fence_loc.append(loc)
    #         fence_setup += f"<DrawBlock x='{xloc}' y='50' z='{zloc}' type='jungle_fence_gate'/> \n"
    #
    #     lava_setup = ""
    #     lava_loc = []
    #     num_of_lava = 5
    #     for i in range(num_of_lava):
    #         xloc = random.randint(-2, 2)
    #         zloc = random.randint(-48, -40)
    #         loc = [xloc, zloc]
    #         while loc in lava_loc or loc in fence_loc:
    #             xloc = random.randint(-2, 2)
    #             zloc = random.randint(-48, -40)
    #             loc = [xloc, zloc]
    #         lava_loc.append(loc)
    #         lava_setup += f"<DrawBlock x='{xloc}' y='49' z='{zloc}' type='lava'/> \n"
    #
    #
    #     for i in trigger_z:
    #         TNT_and_TRIGGERS += f"<DrawBlock x='-2' y='48' z='{i}' type='tnt'/> \n"
    #         TNT_and_TRIGGERS += f"<DrawBlock x='-1' y='48' z='{i}' type='tnt'/> \n"
    #         TNT_and_TRIGGERS += f"<DrawBlock x='-0' y='48' z='{i}' type='tnt'/> \n"
    #         TNT_and_TRIGGERS += f"<DrawBlock x='1' y='48' z='{i}' type='tnt'/> \n"
    #         TNT_and_TRIGGERS += f"<DrawBlock x='2' y='48' z='{i}' type='tnt'/> \n"
    #
    #         TNT_and_TRIGGERS += f"<DrawBlock x='-2' y='50' z='{i}' type='light_weighted_pressure_plate'/> \n"
    #         TNT_and_TRIGGERS += f"<DrawBlock x='-1' y='50' z='{i}' type='light_weighted_pressure_plate'/> \n"
    #         TNT_and_TRIGGERS += f"<DrawBlock x='-0' y='50' z='{i}' type='light_weighted_pressure_plate'/> \n"
    #         TNT_and_TRIGGERS += f"<DrawBlock x='1' y='50' z='{i}' type='light_weighted_pressure_plate'/> \n"
    #         TNT_and_TRIGGERS += f"<DrawBlock x='2' y='50' z='{i}' type='light_weighted_pressure_plate'/> \n"
    #
    #     with open("xml.txt", "w") as f:
    #         f.write(TNT_and_TRIGGERS)
    #
    #     return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    #         <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    #
    #                         <About>
    #                             <Summary>MatureRunner</Summary>
    #                         </About>
    #
    #                         <ServerSection>
    #                             <ServerInitialConditions>
    #                                 <Time>
    #                                     <StartTime>0</StartTime>
    #                                     <AllowPassageOfTime>false</AllowPassageOfTime>
    #                                 </Time>
    #                                 <Weather>clear</Weather>
    #                             </ServerInitialConditions>
    #                             <ServerHandlers>
    #                                 <FlatWorldGenerator generatorString="3;7,2;1;"/>
    #                                 <DrawingDecorator>
    #                                     <DrawCuboid x1='-50' x2='50' y1='50' y2='55' z1='-50' z2='-40' type='air'/>
    #                                     <DrawCuboid x1='-4' x2='-3' y1='49' y2='49' z1='-50' z2='-40' type='stone' />
    #                                     <DrawCuboid x1='3' x2='4' y1='49' y2='49' z1='-50' z2='-40' type='stone' />
    #                                     <DrawCuboid x1='-4' x2='4' y1='49' y2='49' z1='-52' z2='-51' type='stone' />
    #                                     <DrawCuboid x1='-4' x2='4' y1='49' y2='49' z1='-39' z2='-38' type='stone' />
    #                                     <DrawCuboid x1='-2' x2='2' y1='49' y2='49' z1='-50' z2='-40' type='diamond_block'/>''' + \
    #                                 fence_setup + \
    #                                 destination + \
    #                                 lava_setup + \
    #                                 '''</DrawingDecorator>
    #                                 <ServerQuitWhenAnyAgentFinishes/>
    #                                 <ServerQuitFromTimeUp timeLimitMs="1000000"/>
    #                                 <ServerQuitWhenAnyAgentFinishes/>
    #                             </ServerHandlers>
    #                         </ServerSection>
    #
    #                         <AgentSection mode="Survival">
    #                             <Name>Mature Runner</Name>
    #                             <AgentStart>
    #                                 <Placement x="0.5" y="50" z="-49.5" pitch="45" yaw="0"/>
    #                                 <Inventory>
    #                                     <InventoryItem slot="0" type="diamond_pickaxe"/>
    #                                 </Inventory>
    #                             </AgentStart>
    #                             <AgentHandlers>
    #                                 <DiscreteMovementCommands/>
    #                                 <RewardForTouchingBlockType>
    #                                     <Block type='stone' reward='-2'/>
    #                                     <Block type='sandstone' reward='-0.5'/>
    #                                     <Block type='emerald_block' reward='10'/>
    #                                 </RewardForTouchingBlockType>
    #                                 <RewardForCollectingItem>
    #                                     <Item type='bread' reward='2' />
    #                                 </RewardForCollectingItem>
    #                             <ObservationFromFullStats/>
    #                             <ObservationFromRay/>
    #                             <ObservationFromGrid>
    #                                 <Grid name="floorAll">
    #                                     <min x="-'''+str(int(self.obs_size/2))+'''" y="-1" z="-'''+str(int(self.obs_size/2))+'''"/>
    #                                     <max x="'''+str(int(self.obs_size/2))+'''" y="0" z="'''+str(int(self.obs_size/2))+'''"/>
    #                                 </Grid>
    #                             </ObservationFromGrid>
    #                             <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps)+'''" />
    #                             <AgentQuitFromTouchingBlockType>
    #                                 <Block type="emerald_block" />
    #                             </AgentQuitFromTouchingBlockType>
    #                             </AgentHandlers>
    #                         </AgentSection>
    #         </Mission>'''

    # def _get_tnt_and_triggers(self,prepartion_time=25, interval_time=35, step=7):
    #
    #     TNT_and_TRIGGERS = ""
    #     odd = True
    #     num_of_repeater = interval_time  # each repeater will have 0.1 delay
    #     _counter = 0
    #
    #     for i in range((-48 - prepartion_time), -48):
    #         TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='47' z='{i}' type ='stone'/> \n"
    #         TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{i}' face='NORTH' type ='unpowered_repeater'/> \n"
    #     else:
    #         TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{(-48 - prepartion_time)}'  type ='redstone_torch'/> \n"
    #         TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{(-48 - prepartion_time - 1)}'  type ='tnt'/> \n"
    #
    #     for i in range(-48, 50):
    #         if odd:
    #             TNT_and_TRIGGERS += f"<DrawBlock x='{3 + num_of_repeater}' y='47' z='{i}' type ='stone'/> \n"
    #             TNT_and_TRIGGERS += f"<DrawBlock x='{3 + num_of_repeater}' y='48' z='{i}' face ='NORTH' type ='unpowered_repeater'/> \n"
    #             _counter += 1
    #         else:
    #             TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='47' z='{i}' type ='stone'/> \n"
    #             TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{i}' face ='NORTH' type ='unpowered_repeater'/> \n"
    #             _counter -= 1
    #         if odd and _counter == step:
    #             odd = False
    #         elif _counter == 0:
    #             odd = True
    #     else:
    #         odd = True
    #
    #     for i in range(-48, 50, step):
    #         TNT_and_TRIGGERS += f"<DrawBlock x='-1' y='48' z='{i}' type='tnt'/> \n"
    #         TNT_and_TRIGGERS += f"<DrawBlock x='-0' y='48' z='{i}' type='tnt'/> \n"
    #         TNT_and_TRIGGERS += f"<DrawBlock x='1' y='48' z='{i}' type='tnt'/> \n"
    #         # TNT_and_TRIGGERS += f"<DrawBlock x='2' y='48' z='{i}' type='tnt'/> \n"
    #         for j in range(num_of_repeater):
    #             TNT_and_TRIGGERS += f"<DrawBlock x='{3 + j}' y='47' z='{i}' type='stone'/> \n"
    #             if odd:
    #                 TNT_and_TRIGGERS += f"<DrawBlock x='{3 + j}' y='48' z='{i}' face ='WEST' type ='unpowered_repeater'/> \n"
    #             else:
    #                 TNT_and_TRIGGERS += f"<DrawBlock x='{3 + j}' y='48' z='{i}' face ='EAST' type ='unpowered_repeater'/> \n"
    #         else:
    #             TNT_and_TRIGGERS += f"<DrawBlock x='{3 + j + 1}' y='47' z='{i}'  type ='stone'/> \n"
    #             TNT_and_TRIGGERS += f"<DrawBlock x='{3 + j + 1}' y='48' z='{i}'  type ='redstone_wire'/> \n"
    #             TNT_and_TRIGGERS += f"<DrawBlock x='{2}' y='48' z='{i}'  type ='redstone_wire'/> \n"
    #             TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{i}'  type ='redstone_wire'/> \n"
    #         odd = (odd != True)
    #
    #     return TNT_and_TRIGGERS
    #
    # def GetXML(self):
    #     TNT_and_TRIGGERS = self._get_tnt_and_triggers()
    #
    #     with open("xml.txt", "w") as f:
    #         f.write(TNT_and_TRIGGERS)
    #
    #     missionXML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    #                     <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    #
    #                         <About>
    #                             <Summary>Runner</Summary>
    #                         </About>
    #
    #                         <ServerSection>
    #                             <ServerInitialConditions>
    #                                 <Time>
    #                                     <StartTime>0</StartTime>
    #                                     <AllowPassageOfTime>false</AllowPassageOfTime>
    #                                 </Time>
    #                                 <Weather>clear</Weather>
    #                             </ServerInitialConditions>
    #                             <ServerHandlers>
    #                                 <FlatWorldGenerator generatorString="3;7,2;1;"/>
    #                                 <DrawingDecorator>
    #                                     <DrawCuboid x1='-50' x2='50' y1='50' y2='50' z1='-50' z2='50' type='air'/>
    #                                     <DrawCuboid x1='-1' x2='2' y1='49' y2='49' z1='-50' z2='50' type='grass'/>
    #                                     <DrawCuboid x1='-1' x2='2' y1='48' y2='48' z1='-50' z2='50' type='air'/>
    #                                     <DrawCuboid x1='-1' x2='2' y1='47' y2='47' z1='-50' z2='50' type='stone'/>''' + \
    #                  TNT_and_TRIGGERS + \
    #                  '''</DrawingDecorator>
    #                  <ServerQuitWhenAnyAgentFinishes/>
    #                  <ServerQuitFromTimeUp timeLimitMs="1000000"/>
    #                  <ServerQuitWhenAnyAgentFinishes/>
    #              </ServerHandlers>
    #          </ServerSection>
    #
    #          <AgentSection mode="Survival">
    #              <Name>CS175 mature AI demo</Name>
    #              <AgentStart>
    #                  <Placement x="0.5" y="50" z="-49.5" pitch="45" yaw="0"/>
    #                  <Inventory>
    #                      <InventoryItem slot="0" type="diamond_pickaxe"/>
    #                      <InventoryItem slot="1" type="diamond_shovel"/>
    #                      <InventoryItem slot="2" type="repeater"/>
    #
    #                  </Inventory>
    #              </AgentStart>
    #              <AgentHandlers>
    #                  <DiscreteMovementCommands/>
    #
    #                                  <AgentQuitFromTouchingBlockType>
    #                                     <Block type="stone" />
    #                                  </AgentQuitFromTouchingBlockType>
    #                              </AgentHandlers>
    #                          </AgentSection>
    #                      </Mission>'''
    #
    #     return missionXML
    def _get_tnt_and_triggers(slef, length=110, prepartion_time=50, interval_time=15, step=7):
        """
            Generating XML for TNT and redstone triggers
            :Parameterms
                prepartion_time: 10 prepartion time translate into 1 second,
                after 1 second, the first TNT will be ignited, and after 4 second
                the first TNT will explode
                interval_time : The interval time for two rows TNT to be ignited
                 step: number of block between two rows of TNT
                 length : indicating the length of the map
        """
        TNT_and_TRIGGERS = ""
        odd = True
        num_of_repeater = interval_time  # each repeater will have 0.1 delay
        _counter = 0

        for i in range(-prepartion_time, 0):
            TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='47' z='{i}' type ='stone'/> \n"
            TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{i}' face='NORTH' type ='unpowered_repeater'/> \n"
        else:
            TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{(-prepartion_time)}'  type ='redstone_torch'/> \n"
            TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{(-prepartion_time - 1)}'  type ='tnt'/> \n"

        for i in range(0, length):
            if odd:
                TNT_and_TRIGGERS += f"<DrawBlock x='{3 + num_of_repeater}' y='47' z='{i}' type ='stone'/> \n"
                TNT_and_TRIGGERS += f"<DrawBlock x='{3 + num_of_repeater}' y='48' z='{i}' face ='NORTH' type ='unpowered_repeater'/> \n"
                _counter += 1
            else:
                TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='47' z='{i}' type ='stone'/> \n"
                TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{i}' face ='NORTH' type ='unpowered_repeater'/> \n"
                _counter -= 1
            if odd and _counter == step:
                odd = False
            elif _counter == 0:
                odd = True
        else:
            odd = True

        for i in range(0, length, step):
            TNT_and_TRIGGERS += f"<DrawBlock x='-1' y='48' z='{i}' type='tnt'/> \n"
            TNT_and_TRIGGERS += f"<DrawBlock x='-0' y='48' z='{i}' type='tnt'/> \n"
            TNT_and_TRIGGERS += f"<DrawBlock x='1' y='48' z='{i}' type='tnt'/> \n"
            # TNT_and_TRIGGERS += f"<DrawBlock x='2' y='48' z='{i}' type='tnt'/> \n"
            for j in range(num_of_repeater):
                TNT_and_TRIGGERS += f"<DrawBlock x='{3 + j}' y='47' z='{i}' type='stone'/> \n"
                if odd:
                    TNT_and_TRIGGERS += f"<DrawBlock x='{3 + j}' y='48' z='{i}' face ='WEST' type ='unpowered_repeater'/> \n"
                else:
                    TNT_and_TRIGGERS += f"<DrawBlock x='{3 + j}' y='48' z='{i}' face ='EAST' type ='unpowered_repeater'/> \n"
            else:
                TNT_and_TRIGGERS += f"<DrawBlock x='{3 + j + 1}' y='47' z='{i}'  type ='stone'/> \n"
                TNT_and_TRIGGERS += f"<DrawBlock x='{3 + j + 1}' y='48' z='{i}'  type ='redstone_wire'/> \n"
                TNT_and_TRIGGERS += f"<DrawBlock x='{2}' y='48' z='{i}'  type ='redstone_wire'/> \n"
                TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{i}'  type ='redstone_wire'/> \n"
            odd = (odd != True)
        else:  # adding destination block
            TNT_and_TRIGGERS += f"<DrawBlock x='-1' y='49' z='{i + 3}' type='emerald_block'/> \n"
            TNT_and_TRIGGERS += f"<DrawBlock x='-0' y='49' z='{i + 3}' type='emerald_block'/> \n"
            TNT_and_TRIGGERS += f"<DrawBlock x='1' y='49' z='{i + 3}' type='emerald_block'/> \n"
            TNT_and_TRIGGERS += f"<DrawBlock x='2' y='49' z='{i + 3}' type='emerald_block'/> \n"
            TNT_and_TRIGGERS += f"<DrawBlock x='-1' y='49' z='{i + 4}' type='emerald_block'/> \n"
            TNT_and_TRIGGERS += f"<DrawBlock x='-0' y='49' z='{i + 4}' type='emerald_block'/> \n"
            TNT_and_TRIGGERS += f"<DrawBlock x='1' y='49' z='{i + 4}' type='emerald_block'/> \n"
            TNT_and_TRIGGERS += f"<DrawBlock x='2' y='49' z='{i + 4}' type='emerald_block'/> \n"
        return TNT_and_TRIGGERS

    def _get_obstacles(self, obs_density, length):

        assert 0 < obs_density <= 0.3
        obs_types = {1: "jungle_fence_gate"}

        result = ""
        obs_num = int(length * obs_density)
        choices = np.arange(1, length, dtype=np.int32).reshape(-1, 5)
        choices = np.random.permutation(choices)[:obs_num]

        for i in range(choices.shape[0]):
            roll = np.random.choice(list(obs_types.keys()))
            obs = obs_types[roll]
            if type(obs) != list:
                result += f"<DrawBlock x='-1' y='50' z='{choices[i][1]}' type='{obs_types[roll]}'/> \n"
                result += f"<DrawBlock x='0' y='50' z='{choices[i][1]}' type='{obs_types[roll]}'/> \n"
                result += f"<DrawBlock x='1' y='50' z='{choices[i][1]}' type='{obs_types[roll]}'/> \n"
                result += f"<DrawBlock x='2' y='50' z='{choices[i][1]}' type='{obs_types[roll]}'/> \n"
            else:
                for j in range(-1, 3):
                    result += f"<DrawBlock x='{j}' y='50' z='{choices[i][0]}' type='{obs_types[roll][0]}'/> \n"
                    result += f"<DrawBlock x='{j}' y='50' z='{choices[i][1]}' type='{obs_types[roll][1]}'/> \n"
                    result += f"<DrawBlock x='{j}' y='50' z='{choices[i][2]}' type='{obs_types[roll][2]}'/> \n"

        return result

    def GetXML(self):
        """Returns the XML for the project"""
        map_length = 61
        obs_density = 0.3
        TNT_and_TRIGGERS = self._get_tnt_and_triggers(length=map_length)
        obs = self._get_obstacles(obs_density, map_length)

        missionXML = f'''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                        <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                            <About>
                                <Summary>Runner</Summary>
                            </About>
                            <ServerSection>
                                <ServerInitialConditions>
                                    <Time>
                                        <StartTime>0</StartTime>
                                        <AllowPassageOfTime>false</AllowPassageOfTime>
                                    </Time>
                                    <Weather>clear</Weather>
                                </ServerInitialConditions>
                                <ServerHandlers>
                                    <FlatWorldGenerator forceReset="true" generatorString="3;7,2;1;"/>
                                    <DrawingDecorator>
                                        <DrawCuboid x1='-50' x2='50' y1='50' y2='50' z1='0' z2='{map_length}' type='air'/>
                                        <DrawCuboid x1='-1' x2='2' y1='48' y2='48' z1='0' z2='{map_length}' type='air'/>
                                        <DrawCuboid x1='-1' x2='2' y1='49' y2='49' z1='0' z2='{map_length}' type='diamond_block'/>
                                        <DrawCuboid x1='-2' x2='-2' y1='49' y2='50' z1='0' z2='{map_length}' type='stone'/>
                                        <DrawCuboid x1='3' x2='3' y1='49' y2='50' z1='0' z2='{map_length}' type='stone'/>
                                        <DrawCuboid x1='-1' x2='2' y1='49' y2='50' z1='-1' z2='-1' type='stone'/>
                                        <DrawCuboid x1='-1' x2='2' y1='47' y2='47' z1='0' z2='{map_length}' type='stone'/>''' + \
                                        TNT_and_TRIGGERS + \
                                        obs +\
                                        '''</DrawingDecorator>
                                        <ServerQuitWhenAnyAgentFinishes/>
                                        <ServerQuitFromTimeUp timeLimitMs="1000000"/>
                                        <ServerQuitWhenAnyAgentFinishes/>
                                </ServerHandlers>
                            </ServerSection>
                            
                            <AgentSection mode="Survival">
                                <Name>CS175 mature AI demo</Name>
                                <AgentStart>
                                    <Placement x="0.5" y="50" z="0.5" pitch="45" yaw="0"/>
                                    <Inventory>
                                        <InventoryItem slot="0" type="diamond_pickaxe"/>
                                        <InventoryItem slot="1" type="diamond_shovel"/>
                                    </Inventory>
                                </AgentStart>
                                <AgentHandlers>
                                    <DiscreteMovementCommands/>
                                    <RewardForTouchingBlockType>
                                        <Block type='emerald_block' reward='10'/>
                                        <Block type="stone" reward='-1'/>
                                    </RewardForTouchingBlockType>
                                    <RewardForTimeTaken initialReward='0' delta='0.05' density='MISSION_END' />
                                    <ObservationFromFullStats/>
                                    <ObservationFromRay/>
                                    <ObservationFromGrid>
                                        <Grid name="floorAll">
                                            <min x="-'''+str(int(self.obs_size/2))+'''" y="-1" z="-'''+str(int(self.obs_size/2))+'''"/>
                                            <max x="'''+str(int(self.obs_size/2))+'''" y="0" z="'''+str(int(self.obs_size/2))+'''"/>
                                        </Grid>
                                    </ObservationFromGrid>
                                    <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps)+'''" />
                                    <AgentQuitFromTouchingBlockType>
                                        <Block type="emerald_block" />
                                    </AgentQuitFromTouchingBlockType>
                                </AgentHandlers>
                            </AgentSection>
                        </Mission>'''

        return missionXML

if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=MinecraftRunner, config={
        'env_config': {},  # No environment parameters to configure
        'framework': 'torch',  # Use pyotrch instead of tensorflow
        'num_gpus': 0,  # We aren't using GPUs
        'num_workers': 0  # We aren't using parallelism
    })

    while True:
        print(trainer.train())
