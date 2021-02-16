from numpy.random import randint
import numpy as np
import random


OBS_SIZE = 11
MAX_EPISODE_STEPS = 200

def _get_tnt_and_triggers(length=110, prepartion_time=50, interval_time=15, step=7):
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


def _get_obstacles(obs_density, length, difficulty= 0):
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

def Map():
    """Returns the XML for the project"""
    map_length = 22
    assert (map_length -2) % 5 == 0
    obs_density = 0.3
    TNT_and_TRIGGERS = _get_tnt_and_triggers(length=map_length)
    obs = _get_obstacles(obs_density, map_length)

    sandstone_setup = ""
    sandstone_loc = []

    zloc_list = [i for i in range(2, map_length, 3)]
    for zloc in zloc_list:
        xloc = random.randint(-1, 1)
        loc = [xloc, zloc]
        sandstone_loc.append(loc)
        sandstone_setup += f"<DrawBlock x='{xloc}' y='50' z='{zloc}' type='sandstone' /> \n"
        sandstone_setup += f"<DrawBlock x='{xloc+ 1}' y='50' z='{zloc}' type='sandstone' /> \n"


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
                                    obs + \
                                '''</DrawingDecorator>
                                <ServerQuitWhenAnyAgentFinishes/>
                                <ServerQuitFromTimeUp timeLimitMs="100000"/>
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
                                <RewardForCollectingItem>
                                    <Item type='diamond' reward='1'/>
                                </RewardForCollectingItem>
                                <RewardForTimeTaken initialReward='0' delta='0.05' density='MISSION_END' />
                                <ObservationFromFullStats/>
                                <ObservationFromRay/>
                                <ObservationFromGrid>
                                    <Grid name="floorAll">
                                        <min x="-''' + str(int(OBS_SIZE/2)) + '''" y="-1" z="-''' + str(int(OBS_SIZE/2)) + '''"/>
                                        <max x="''' + str(int(OBS_SIZE/2)) + '''" y="0" z="''' + str(int(OBS_SIZE/2)) + '''"/>
                                    </Grid>
                                </ObservationFromGrid>
                                <AgentQuitFromReachingCommandQuota total="''' + str(MAX_EPISODE_STEPS) + '''" />
                                <AgentQuitFromTouchingBlockType>
                                    <Block type="emerald_block" />
                                    </AgentQuitFromTouchingBlockType>
                            </AgentHandlers>
                        </AgentSection>
                    </Mission>'''

    return missionXML
