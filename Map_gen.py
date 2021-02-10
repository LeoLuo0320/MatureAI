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

def _get_tnt_and_triggers(prepartion_time = 25,interval_time = 35,step = 7):

    TNT_and_TRIGGERS = ""
    odd = True
    num_of_repeater = interval_time  # each repeater will have 0.1 delay
    _counter = 0

    for i in range((-48-prepartion_time),-48):
        TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='47' z='{i}' type ='stone'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{i}' face='NORTH' type ='unpowered_repeater'/> \n"
    else:
        TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{(-48-prepartion_time)}'  type ='redstone_torch'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='{3}' y='48' z='{(-48-prepartion_time-1)}'  type ='tnt'/> \n"

    for i in range(-48,50):
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


    for i in range(-48, 50, step):
        TNT_and_TRIGGERS += f"<DrawBlock x='-1' y='48' z='{i}' type='tnt'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='-0' y='48' z='{i}' type='tnt'/> \n"
        TNT_and_TRIGGERS += f"<DrawBlock x='1' y='48' z='{i}' type='tnt'/> \n"
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

    return TNT_and_TRIGGERS




def GetXML():
    TNT_and_TRIGGERS = _get_tnt_and_triggers()

    with open("xml.txt", "w") as f:
        f.write(TNT_and_TRIGGERS)

    missionXML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
                                <FlatWorldGenerator generatorString="3;7,2;1;"/>
                                <DrawingDecorator>
                                    <DrawCuboid x1='-50' x2='50' y1='50' y2='50' z1='-50' z2='50' type='air'/>
                                    <DrawCuboid x1='-1' x2='2' y1='49' y2='49' z1='-50' z2='50' type='grass'/>
                                    <DrawCuboid x1='-1' x2='2' y1='48' y2='48' z1='-50' z2='50' type='air'/>
                                    <DrawCuboid x1='-1' x2='2' y1='47' y2='47' z1='-50' z2='50' type='stone'/>''' + \
                 TNT_and_TRIGGERS + \
                 '''</DrawingDecorator>
                 <ServerQuitWhenAnyAgentFinishes/>
                 <ServerQuitFromTimeUp timeLimitMs="1000000"/>
                 <ServerQuitWhenAnyAgentFinishes/>
             </ServerHandlers>
         </ServerSection>

         <AgentSection mode="Survival">
             <Name>CS175 mature AI demo</Name>
             <AgentStart>
                 <Placement x="0.5" y="50" z="-49.5" pitch="45" yaw="0"/>
                 <Inventory>
                     <InventoryItem slot="0" type="diamond_pickaxe"/>
                     <InventoryItem slot="1" type="diamond_shovel"/>
                     <InventoryItem slot="2" type="repeater"/>

                 </Inventory>
             </AgentStart>
             <AgentHandlers>
                 <ContinuousMovementCommands/>
                 <VideoProducer>
                     <Width>''' + str(video_width) + '''</Width>
                                     <Height>''' + str(video_height) + '''</Height>
                                 </VideoProducer>
                                 <AgentQuitFromTouchingBlockType>
                                    <Block type="stone" />
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
