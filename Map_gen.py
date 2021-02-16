from builtins import range
import MalmoPython
import os
import sys
import time
from numpy.random import randint
import numpy as np

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

video_width = 800
video_height = 600

def _get_tnt_and_triggers(length = 110,prepartion_time = 50,interval_time = 15,step = 7):
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

    for i in range(-prepartion_time,0):
        TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='47' z='{i}' type ='stone'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{i}' face='NORTH' type ='unpowered_repeater'/> \n"
    else:
        TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{(-prepartion_time)}'  type ='redstone_torch'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{(-prepartion_time-1)}'  type ='tnt'/> \n"

    for i in range(0,length):
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
        # TNT_and_TRIGGERS += f"<DrawBlock x='-1' y='48' z='{i}' type='tnt'/> \n"
        # TNT_and_TRIGGERS += f"<DrawBlock x='-0' y='48' z='{i}' type='tnt'/> \n"
        # TNT_and_TRIGGERS += f"<DrawBlock x='1' y='48' z='{i}' type='tnt'/> \n"
        # TNT_and_TRIGGERS += f"<DrawBlock x='2' y='48' z='{i}' type='tnt'/> \n"
        for j in range(num_of_repeater):
            TNT_and_TRIGGERS += f"<DrawBlock x='{3+j}' y='47' z='{i}' type='stone'/> \n"
            if odd:
                TNT_and_TRIGGERS += f"<DrawBlock x='{3+j}' y='48' z='{i}' face ='WEST' type ='unpowered_repeater'/> \n"
            else:
                TNT_and_TRIGGERS += f"<DrawBlock x='{3+j}' y='48' z='{i}' face ='EAST' type ='unpowered_repeater'/> \n"
        else:
            TNT_and_TRIGGERS += f"<DrawBlock x='{3 + j+1}' y='47' z='{i}'  type ='stone'/> \n"
            TNT_and_TRIGGERS += f"<DrawBlock x='{3 + j+1}' y='48' z='{i}'  type ='redstone_wire'/> \n"
            TNT_and_TRIGGERS += f"<DrawBlock x='{2}' y='48' z='{i}'  type ='redstone_wire'/> \n"
            TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{i}'  type ='redstone_wire'/> \n"
        odd = (odd != True)
    else: # adding destination block
        TNT_and_TRIGGERS += f"<DrawBlock x='-1' y='49' z='{i+3}' type='emerald_block'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='-0' y='49' z='{i+3}' type='emerald_block'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='1' y='49' z='{i+3}' type='emerald_block'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='2' y='49' z='{i+3}' type='emerald_block'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='-1' y='49' z='{i+4}' type='emerald_block'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='-0' y='49' z='{i+4}' type='emerald_block'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='1' y='49' z='{i+4}' type='emerald_block'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='2' y='49' z='{i+4}' type='emerald_block'/> \n"
    return TNT_and_TRIGGERS

def _get_obstacles(obs_density, length, difficulty= 1):
    '''
    :param obs_density: density of obstacle given the length of the map
    :param length: used by obs_density
    :param difficulty: 0 means easy, nonezeros will have stone_slab and fence
    '''
    assert 0 < obs_density <= 0.3
    obs_types = {1: "jungle_fence_gate"} if difficulty == 0 else {1: "jungle_fence_gate", 2: ["stone_slab", "fence"]}

    result = ""
    obs_num = int(length * obs_density)
    choices = np.arange(2, length, dtype=np.int32).reshape(-1, 5)
    choices = np.random.permutation(choices)[:obs_num]

    for i in range(choices.shape[0]):
        diamon_placement = np.random.choice([-1, 0 , 1, 2])
        roll = np.random.choice(list(obs_types.keys()))
        obs = obs_types[roll]
        if type(obs) != list:
            row = np.random.choice(choices[i])  # we will spawn gate from range(0 - 4)
            result += f"<DrawBlock x='-1' y='50' z='{row}' type='{obs_types[roll]}'/> \n"
            result += f"<DrawBlock x='0' y='50' z='{row}' type='{obs_types[roll]}'/> \n"
            result += f"<DrawBlock x='1' y='50' z='{row}' type='{obs_types[roll]}'/> \n"
            result += f"<DrawBlock x='2' y='50' z='{row}' type='{obs_types[roll]}'/> \n"
            result += f"<DrawItem x='{diamon_placement}' y='50' z='{row+1}' type='diamond' /> \n"

        else:
            fence_gap = np.random.choice([0, 1, 2])
            for j in range(-1, 3):
                result += f"<DrawBlock x='{j}' y='50' z='{choices[i][0]}' type='{obs_types[roll][0]}'/> \n"
                result += f"<DrawBlock x='{j}' y='50' z='{choices[i][1] + fence_gap}' type='{obs_types[roll][1]}'/> \n"
                result += f"<DrawItem x='{diamon_placement}' y='50' z='{choices[i][1] + fence_gap + 1}' type='diamond' /> \n"
    return result

def GetXML(obs_size =5):
    """Returns the XML for the project"""
    map_length = 51
    assert (map_length -2) % 5 == 0

    obs_density = 0.3
    TNT_and_TRIGGERS = _get_tnt_and_triggers(length=map_length)
    obs = _get_obstacles(obs_density, map_length)

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
                                    <DrawCuboid x1='-1' x2='2' y1='47' y2='47' z1='0' z2='{map_length}' type='stone'/>'''  +\
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
                                            <min x="-'''+str(int(obs_size/2))+'''" y="-1" z="-'''+str(int(obs_size/2))+'''"/>
                                            <max x="'''+str(int(obs_size/2))+'''" y="0" z="'''+str(int(obs_size/2))+'''"/>
                                        </Grid>
                                    </ObservationFromGrid>
                                    <AgentQuitFromReachingCommandQuota total="'''+"1000"+'''" />
                                    <AgentQuitFromTouchingBlockType>
                                        <Block type="emerald_block" />
                                    </AgentQuitFromTouchingBlockType>
                                </AgentHandlers>
                            </AgentSection>
                        </Mission>'''

    return missionXML


if __name__ == "__main__":
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    my_mission = MalmoPython.MissionSpec(GetXML(), True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    # Attempt to start a mission:
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_mission_record)
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)

    print("\nMission running ", end=' ')

    # Loop until mission ends:
    while world_state.is_mission_running:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)

    print("\nMission ended")
