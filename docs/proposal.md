## 1. Summary of the Project 

#### 1.1 Setup and Goals

In this project, our team is planning to develop an intelligent agent under Temple Run like setting. The goal of this AI is to collect as many points (diamonds/gold) as possible, while trying to escape from the ghost. There are different kind of challenges like wide gap, obstacles on the top and dip along the way. Thus, we plan to implement a random map generator combining different obstacles, which also supports seed functionality.

| <div align="center">Reference Picture</div>                  |
| ------------------------------------------------------------ |
| <div align="center"><img src="https://cdn.slashgear.com/wp-content/uploads/2012/02/TempleRun-screens.jpeg" alt="Temple Run for Android to be announced via Facebook - SlashGear" style="zoom: 50%;" /></div> |

#### 1.2 Possible Input&Output

The output of our AI will combination of move right/left, jump, and use items/potions.

The input of our agent could be raw pixels on the screen, or the blocks that is close to the agent.

## 2. Candidate Algorithms

1. Reinforcement learning with neural function
2. A* search
3. Deep learning for images

## 3. Evaluation Plan

#### 3.1 Quantitative Evaluation

<strong>Numeric</strong>

For each random map generated, we calculted the regret metric. We will evaluate the difference bewteen the reward, gold and diamond in this scenario, of optimal decision and that collected by the agent. The smaller the difference, the better the performance of our agent. 

<strong>Baselines</strong>

The goal of our agent is to jump across wide gaps and escape from the ghost, or it will die and the game will end. At the same time, it should also collect as much gold and diamond as possible. 

#### 3.2 Qualitative Evaluation

<strong>Simple Example Cases</strong>

Our agent are expected to survive as long as possible, which means that it must jump across obstacles and turn left or right to avoid hitting the wall. At the same time, it will collect gold and diamond as much as possible, without taking greate risk of dying. 

<strong>Error Analysis and Introspection</strong>

For unexpected behavior and wired output, we will track actions of our agent and look at the generated map again to conclude what is the issue. If it does not work, we can output the map and action list, then hand simulate the process to find potential improvement. 

<strong>The Super-Impressive Example</strong>

Our agent will never die and it will run forever! At the same time, it can discern traps. For example, he will not collect gold above a gap, because it will die for that. Also, it will jump to collect gold or diamond if there is any. 

## 4. Appointment with the Instructor

