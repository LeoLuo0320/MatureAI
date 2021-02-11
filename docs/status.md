---
layout: default
title: Status
---

# Project Summary
MatureAI is a survival game. The goal of this project is training our agent to survive as long as possible and collect as many rewards as it can. We are inspired by the game Temple Run and would like to design a much more complex AI runner project with the help of deep Q-Learning algorithm. 
Our agent needs to collect golds and diamonds while running. When collecting rewards, it also needs to bypass obstacles and not to hit stones on both sides of the road. In the future, we will use Redstone circuitry to create explosion and destroy the road as punishment. At the same time, we will give our agent surviving time reward for moving forward.

# Approach

# Evaluation
To evaluate our model quantitatively, we set positive and negative rewards for touching specific blocks. We will evaluate the performance of our agent based on the cumulative value returned by our agent. Please see the code snip for details. 
```
<RewardForTouchingBlockType>
  <Block type='stone' reward='-1'/>
  <Block type='sandstone' reward='-0.5'/>
  <Block type='emerald_block' reward='10'/>
  <Block type='jungle_fence_gate' reward='-1'/>
</RewardForTouchingBlockType>
```
Based on the logic designed above, we have the reward graph below. According to the graph, we could see that our agent is learning to grab golds and diamonds and avoid hitting obstacles. 

<img width="700" alt="reward-graph" src="https://need_graph_here">

To evaluate our model Qualitatively, we will keep track of the action taken by our agent. It is expected to survive as long as possible, which means that our agent should bypass obstacles and avoid hitting stones on boths sides of the road, which will trigger TNT in the future version. At the same time, it will collect gold and diamond as much as possible, without taking greate risk of dying.

# Remaining Goals and Challenges

## Remaining Goals
In the next four weeks, we will change discrete action space to continuous action space as our environment will get more complex. For now, we make a simple map to ensure our agent will avoid hitting obstacles and collect golds. In the final version, we will design more complex maps for the agent and add more types of obstacles. 
At the same time, our ambition is to use raw pixel instead of world state observation. This requires color segmentation and distance estimation, which are the main challenge we face in the next few weeks. 

## Challenges
When we try to use computer vision and raw pixels, we realize that it is more complex than we initially thought. 
The first challenge is using computer vision to detect surrounding objects. We will implement a color segmentation method to quickly detect the boundary of each object, and separate them using different colors. We need to be really careful when setting threshold values, because colors of some single blocks are composed of multiple colors, and we need to treat them as a whole and replace them with the same color.
The second challenge is to use computer vision to detect obstacles. When trying to implement computer vision with opencv, we have to find the distance obstacles and our agent, then reflect this distance in our map. This step requires careful calculation so our agent will know exactly where the obstacles lie. 

# Resources Used
- [OpenCV](https://opencv.org/)
- Malmo Tutorial
- Assignment 2
- Stack Overflow
