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
