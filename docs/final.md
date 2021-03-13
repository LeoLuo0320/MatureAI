---
layout: default
title: Final Report
---

# Video Summary


# 1.Project Summary
Our project MatureAI is a survival game. Our map is composed of a 4 blocks wide running track surrounded by dark oak fences. Rewards and obstacles are randomly generated for each round. The goal of our agent is to survive as long as possible, to collect diamonds when moving forward, and to reach the target location. Depending on the obstacle, our agent learns to take appropriate actions, such as opening the gate, stepping on the stone and jumping over the fence. The agent is dropped at the start line of the track for each game, and we use Redstone circuitry to create explosions and destroy the road as time goes by, so the agent learns to move forward and reach the finish line, or it will die. To improve the performance of the agent, we customized the PPO trainer with PyTorch CNN model and optimized our reward function. Compared to the status report, the map is more complex, our agent bypasses more obstacles and survives much longer.



# 2. Learning Environment

### 2.1 Environment Summary 

Compared to the status report, we have a huge update in the final version. In the status report our environment is simple and deterministic, but for the final report we changed the map into more complex and stochastic environment. In the status report we planned to implement four levels of difficulty, but because of the limitation of Malmo platform (there is no dashing action for the agent) we only train our agent on the introductory level. To compensate the change, we added some interesting creatures and map generating mechanism, such as using continuous action space and random reward distribution. 

### 2.2 Obstacle Types

##### **1 Road Destruction(Difficulty: easy, Deterministic)**

<p align="center">
<img width="350" src=".\img\tnt.gif">
</p>

The initially the agent will have 6 second to run before the first TNT explodes. 

##### **2 Simple Jumping( Difficulty: easy, Deterministic)**

<p align="center">
<img width="350" src=".\img\jump_fence.gif">
</p>

The agent needs to step onto the slab, perform the jump action, and walk through the gate.

##### **3 Opening Door (Difficulty: medium, Stochastic)**

<p align="center">
<img width="350" src=".\img\door_open.gif">
</p>

The agent needs to perform 'open action', and immediately perform 'stop action' and walk through the gate. There will be only two doors generated randomly for the agent to open and the other two are fences that the agent need to move the corresponding gate and open it.

##### **4 Avoiding Fireball (Difficulty: hard, Stochastic)**

<p align="center">
<img width="350" src=".\img\ghost.gif">
</p>

The agent needs to avoid the fireballs that the ghost shoots, and also the fire after the explosion. Because our obstacles are made of wood, the fire will ignite the fences and the agent needs to avoid those as well. 

##### **5 Collecting Rewards(Difficulty: medium, Stochastic)**

Behind each types of obstacles, our map will distribute diamond randomly as reward. The agent needs to perform the correct action and claim the reward as soon as possible because of the following explosive and the fire balls will burn the reward. 




# 3. Approaches

### Approach 1: Customize PPO Trainer
Compared to the status report, we customized PPO trainer with CNN network instead of the default model to let the agent learn spatial information of the environment. In our customized trainer class, we use PyTorch library and add three convolution layers to extract features from observation matrices. As our input matrices are not large, we use outputs from convolution layers without adding pooling layers in between and use RELU provided by PyTorch as the activation function. Compared to using linear function with default PPO trainer, our agent learns faster and more accurate under same number of steps.

```
 class MyModel(TorchModelV2, nn.Module):
     def __init__(self, *args, **kwargs):
         TorchModelV2.__init__(self, *args, **kwargs)
         nn.Module.__init__(self)

         self.conv1 = nn.Conv2d(4, 32, kernel_size=7, padding=3)
         self.conv2 = nn.Conv2d(32, 32, kernel_size=7, padding=3)
         self.conv3 = nn.Conv2d(32, 32, kernel_size=7, padding=3)
         self.policy_layer = nn.Linear(32*15*15, 5)
         self.value_layer = nn.Linear(32*15*15, 1)
         self.value = None

     def forward(self, input_dict, state, seq_lens):
         x = input_dict['obs']
         x = F.relu(self.conv1(x))
         x = F.relu(self.conv2(x))
         x = F.relu(self.conv3(x))
         x = x.flatten(start_dim=1)

         policy = self.policy_layer(x)
         self.value = self.value_layer(x)

         return policy, state

     def value_function(self):
         return self.value.squeeze(1)
```

(Maybe use flow chart instead of code)

### Approach 2: Optimize Reward 

#### Adding Reward for Approaching destination

One issue we have for the status report is that the agent sometimes moved around instead of moving forward, and it was finally killed by TNT bombs because of staying in the same area for too long. To resolve this issue and improve the performance of our agent, we add positive rewards for approaching the destination. This reward helps our agent learn to move forward and reach the finish line with less undesirable situations, such as moving around in circles and jumping off the boundary. 

```
 old_dest = self.current_to_dest  # Used for giving reward of moving to the destination
 old_shortest = self.shortest_to_dest
 new_dest = self.current_to_dest
 new_shortest = self.shortest_to_dest
 if old_dest < new_dest:
     reward -= 0.5
 elif old_dest > new_dest:
     reward += 0.5

 if old_shortest < new_shortest:
     reward -= 1
 elif old_shortest > new_shortest:
     reward += 1

 self.episode_return += reward
```

First, our project will compare the z value of the current round and that from the last round. Whenever this shortest distance is updated, we give it +1 reward since it means the agent is moving towards the destination. On the other hand, if the agent is moving towards the opposite direction, we give it -0.5 reward. We do not give it -1 reward since moving towards the opposite direction may not always be a bad thing since it may on its way towards the diamond. From the evaluation result, we conclude that this reward undoubtfully contributes to improving survival time of our agent. 

#### Reward Summury
For the final version, we consider 5 factors when giving our agent rewards. Those factors are survival time in seconds, if the distance to the destination is smaller, collecting diamonds, reaching walls, and reaching the destination. The following is the reward formula. 

    R(s) = 1 * SurvivalTime + 1 * CloserToDest – 0.5* FartherToDest + 1* CollectingDiamonds -  1 * ReachWalls + 10 * ReachDest

As a survival game, it is intuitive to use survival time as rewards. We use “RewardForTimeTaken” in the XML documentation to give reward to agent by counting the time it survives. Since one tick in Minecraft is 0.05s in real world, we give 0.05 reward for every tick it survives in the game, which is the same as +1 reward per second.

Then, we want the agent to move in the right direction, i.e. towards the destination, while still be able to make turns since it should bypass the obstacles and collect diamonds. So we have rewards such as “CloserToDest”, “FartherToDest”, “ReachWalls”. For “CloserToDest” and “FartherToDest”, we record the shortest distance to the destination when the agent moves. For “ReachWalls”, we check if the agent is touching the wall’s type, which is “dark_oak_fence”. If so, the agent gets -1 reward since moving to the wall is just wasting time and the agent may die. 

Finally, we give the agent +1 reward whenever it collects the diamond and +10 reward when it reaches the destination.



# 4 Evaluation


# 5 Resources Used
- [RLlib](https://docs.ray.io/en/master/rllib-training.html)
- [Pytorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Customized RLlib Video](https://youtu.be/nMzoYNHgLpY)
- [Malmo API Documentation](https://microsoft.github.io/malmo/0.30.0/Documentation/index.html)
- [Malmo Tutorial](http://microsoft.github.io/malmo/0.30.0/Python_Examples/Tutorial.pdf)
- [OpenCV](https://opencv.org/)
- [Image Segmentation in OpenCV](https://realpython.com/python-opencv-color-spaces/)
- Assignment 2
