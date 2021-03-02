from numpy.random import randint
import numpy as np
import random

OBS_SIZE = 15
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
        TNT_and_TRIGGERS += f"<DrawBlock x='-1' y='49' z='{i + 2}' type='emerald_block'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='-0' y='49' z='{i + 2}' type='emerald_block'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='1' y='49' z='{i + 2}' type='emerald_block'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='2' y='49' z='{i + 2}' type='emerald_block'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='-1' y='49' z='{i + 3}' type='emerald_block'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='-0' y='49' z='{i + 3}' type='emerald_block'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='1' y='49' z='{i + 3}' type='emerald_block'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='2' y='49' z='{i + 3}' type='emerald_block'/> \n"
    return TNT_and_TRIGGERS


def _get_obstacles(obs_density, length, difficulty=0):
    '''
    :param obs_density: density of obstacle given the length of the map
    :param length: used by obs_density
    :param difficulty: 0 means easy, nonezeros will have stone_slab and fence with gap 0/1
                        , value greater than 1 will have Ghast shotting fire balls
    '''
    assert 0 < obs_density <= 0.3
    DIAMOND_POS = []
    result = ""
    if difficulty > 1:
        for i in range(2, length, 40):  # for every 30 blocks we will spawn a ghast
            result += f"<DrawEntity x='-15' y='65' z='{i}' type='Ghast'/>"
    obs_types = {1: "fence_gate"} if difficulty == 0 else {1: "fence_gate",
                                                                  2: ["stone_slab", "acacia_fence"]}
    obs_num = int(length * obs_density)
    choices = np.arange(2, length, dtype=np.int32).reshape(-1, 5)
    choices = np.random.permutation(choices)[:obs_num]

    for i in range(choices.shape[0]):
        diamon_placement = np.random.choice([-1, 0, 1, 2])
        roll = np.random.choice(list(obs_types.keys()))
        enable_col = np.random.choice([-1, 0, 1, 2], 2, replace=False)
        obs = obs_types[roll]
        if type(obs) != list:
            row = np.random.choice(choices[i][1:])  # we will spawn gate from range(1 - 4)
            for j in enable_col:
                result += f"<DrawBlock x='{j}' y='50' z='{row}' type='{obs_types[roll]}'/> \n"
            for j in np.setdiff1d([-1, 0, 1, 2], enable_col):  # disabled gate we will use fence instead
                result += f"<DrawBlock x='{j}' y='50' z='{row}' type='fence'/> \n"
            result += f"<DrawItem x='{diamon_placement}' y='50' z='{row + 1}' type='diamond' /> \n"
            DIAMOND_POS.append((diamon_placement, row + 1))
        else:
            fence_gap = 0
            for j in range(-1, 3):
                result += f"<DrawBlock x='{j}' y='50' z='{choices[i][0]}' type='{obs_types[roll][0]}'/> \n"
                result += f"<DrawBlock x='{j}' y='50' z='{choices[i][1] + fence_gap}' type='{obs_types[roll][1]}'/> \n"
            result += f"<DrawItem x='{diamon_placement}' y='50' z='{choices[i][1] + fence_gap + 1}' type='diamond' /> \n"
            DIAMOND_POS.append((diamon_placement, choices[i][1] + fence_gap + 1))
    return result, DIAMOND_POS


def Map():
    """Returns the XML for the project"""
    map_length = 22
    assert (map_length - 2) % 5 == 0
    obs_density = 0.3
    TNT_and_TRIGGERS = _get_tnt_and_triggers(length=map_length)
    obs, DIAMOND_POS = _get_obstacles(obs_density, map_length, difficulty=2)

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
                                    <DrawCuboid x1='-30' x2='10' y1='80' y2='80' z1='-6' z2='{map_length}' type='glass'/>
                                    <DrawCuboid x1='-30' x2='-30' y1='80' y2='0' z1='-6' z2='{map_length}' type='glass'/>
                                    <DrawCuboid x1='-30' x2='10' y1='80' y2='0' z1='-6' z2='-6' type='glass'/>
                                    <DrawCuboid x1='-30' x2='8' y1='80' y2='0' z1='{map_length + 3}' z2='{map_length + 3}' type='glass'/>
                                    <DrawCuboid x1='-30' x2='10' y1='10' y2='10' z1='-6' z2='{map_length}' type='glass'/>
                                    <DrawCuboid x1='10' x2='10' y1='80' y2='10' z1='-6' z2='{map_length+2}' type='diamond_block'/>
                                    <DrawCuboid x1='-1' x2='2' y1='47' y2='47' z1='0' z2='{map_length}' type='stone'/>
                                    <DrawCuboid x1='-1' x2='2' y1='49' y2='49' z1='-1' z2='{map_length}' type='diamond_block'/> 
                                    <DrawCuboid x1='3' x2='3' y1='50' y2='50' z1='-1' z2='{map_length}' type='dark_oak_fence'/>
                                    <DrawCuboid x1='-2' x2='-2' y1='50' y2='50' z1='-1' z2='{map_length}' type='dark_oak_fence'/>
                                    <DrawCuboid x1='-1' x2='2' y1='50' y2='50' z1='-1' z2='-1' type='dark_oak_fence'/>
                                    ''' + \
                 TNT_and_TRIGGERS + \
                 obs + \
                 '''</DrawingDecorator>
                 <ServerQuitWhenAnyAgentFinishes/>
                 <ServerQuitFromTimeUp timeLimitMs="100000"/>
             </ServerHandlers>
         </ServerSection>
         <AgentSection mode="Survival">
             <Name>CS175 mature AI demo</Name>
             <AgentStart>
                 <Placement x="0.5" y="50" z="0.5" pitch="45" yaw="0"/>
             </AgentStart>
             <AgentHandlers>
                 <ContinuousMovementCommands/>
                 <ChatCommands/>
                 <RewardForTouchingBlockType>
                     <Block type='emerald_block' reward='10'/>
                     <Block type="glass" reward='-2'/>
                     <Block type="dark_oak_fence" reward='-1'/>
                     <Block type="fire" reward='-1'/>
                 </RewardForTouchingBlockType>
                 <RewardForCollectingItem>
                     <Item type='diamond' reward='1'/>
                 </RewardForCollectingItem>
                 <RewardForTimeTaken initialReward='0' delta='0.05' density='MISSION_END' />
                 <ObservationFromFullStats/>
                 <ObservationFromRay/>
                 <ObservationFromGrid>
                     <Grid name="floorAll">
                         <min x="-''' + str(int(OBS_SIZE / 2)) + '''" y="0" z="-''' + str(int(OBS_SIZE / 2)) + '''"/>
                                        <max x="''' + str(int(OBS_SIZE / 2)) + '''" y="0" z="''' + str(int(OBS_SIZE / 2)) + '''"/>
                                    </Grid>
                                </ObservationFromGrid>
                                <AgentQuitFromReachingCommandQuota total="''' + str(MAX_EPISODE_STEPS) + '''" />
                                <AgentQuitFromTouchingBlockType>
                                    <Block type='emerald_block'/>
                                </AgentQuitFromTouchingBlockType>
                            </AgentHandlers>
                        </AgentSection>
                    </Mission>'''

    return missionXML, DIAMOND_POS