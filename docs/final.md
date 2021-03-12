---
layout: default
title: Final Report
---

# Video Summury


# Project Summary
Our project MatureAI is a survival game. Our map is composed of a 4 blocks wide running track that is surrounded by dark oak fences. Rewards and obstacles are randomly generated for each round. The goal of our agent is to survive as long as possible, collect diamonds when moving forward, and reach the target location. When collecting diamonds, our agent needs to bypass and avoid obstacles such as stones, fences, and gates. Depending on these obstacles, our agent learns to take appropriate actions, such as opening the gate, step on the stone and jump over the fence. The agent is dropped at the start line of the track for each game, and we use Redstone circuitry to create explosions and destroy the road as time goes by, so the agent learns to move forward and reach the finish line, or it will die.

Compared to the status report, we have a huge update in the final version. We designed 4 levels of difficulty: trainee, introductory, medium, and advance. For the status report, we only handled the trainee level, which uses discrete action space and only contains one obstacle, i.e., fence gates. Our agent only needs to open the gate and walk through the gate. For the final version, we implemented the advanced level and changed our map dramatically to make our project more complex and interesting. Firstly, we used continuous action space, which is more challenging and requires more precise actions compared to discrete action space. For the obstacles, apart from using fence gates, we added Acacia fences, slab stones, and diamonds to our map. To bypass acacia fences, our agent needs to step onto the slab stone and jump over the fence. Sometimes the agent will find the track is blocked by a combination of fences and fence gates, so our agent needs to discern the fence gate and open it to pass this blocking line. Also, the agent will get extra reward if it collects diamonds when it is bypassing the obstacles. Collecting the diamond is not necessary for survival. On the other hand, the agent may even die because of collecting the diamond since it may not have enough time to move forward and bypassing the obstacles. To improve the performance of the agent, we customized the PPO trainer with PyTorch CNN model and optimized our reward function. Compared to the status report, the map is more complex, our agent bypasses more obstacles and survives much longer.


# Approaches

### Approach 1: Customize PPO Trainer
Compared to the status report, we customized PPO trainer with CNN network with the video following the tutorial video provided on Campuswire. In our customized trainer class, we use PyTorch library and add three convolution layers to extract features from observation matrices. As our input matrices are not large, we use outputs from convolution layers without adding pooling layers in between, and use RELU provided by PyTorch as the activation function. Compared to using linear function with default PPO trainer, our agent learns faster and more accurate under same number of steps. 

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

Then, we want the agent to move in the right direction, i.e. towards the destination, while still be able to make turns since it should bypass the obstacles and collect diamonds. So we have rewards such as “CloserToDest”, “FartherToDest”, “ReachWalls”. For “CloserToDest” and “FartherToDest”, we record the shortest distance to the destination when the agent moves. For “ReachWalls”, we check if the agent is touching the wall’s type, which is “dark_oak_fence”. If so, the agent gets -1 reward since moving to the wall is just wasting time and the agent may die. Finally, we give the agent +1 reward whenever it collects the diamond and +10 reward when it reaches the destination.



# Evaluation


# Resources Used
- [RLlib](https://docs.ray.io/en/master/rllib-training.html)
- [Pytorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Customized RLlib Video](https://youtu.be/nMzoYNHgLpY)
- [Malmo API Documentation](https://microsoft.github.io/malmo/0.30.0/Documentation/index.html)
- [Malmo Tutorial](http://microsoft.github.io/malmo/0.30.0/Python_Examples/Tutorial.pdf)
- [OpenCV](https://opencv.org/)
- [Image Segmentation in OpenCV](https://realpython.com/python-opencv-color-spaces/)
- Assignment 2
