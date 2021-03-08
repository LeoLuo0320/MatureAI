---
layout: default
title: Final Report
---

# Video Summury


# Project Summary
Our project MatureAI is a survival game. Our map is composed of a 4 blocks wide running track that is surrounded by dark oak fences, and rewards and obstacles are randomly generated for each round. The goal of our agent is to survive as long as possible, collect diamonds when moving forward, and reach the target location. When collecting diamonds, our agent needs to bypass and avoid obstacles, which include fire, stones, fences, and gates. Depending on these obstacles, our agent learns to take appropriate actions, such as opening the gate, step on the stone and jump over the fence. The agent is dropped at the start line of the track for each game, and we use Redstone circuitry to create explosions and destroy the road as time goes by, so the agent learns to move forward and reach the finish line, or it will die.

Compared to the status report, we have a huge update in the final version. We designed 4 levels of difficulty: trainee, introductory, medium, and advance. For the status report, we only handled the trainee level, which only contains fence gates, and our agent only needs to open the gate and walk through the gate. For the final version, we implemented the advanced level and changed our map dramatically to make our project more complex and look better. Apart from using fence gates, we added Acacia fences, slab stones, and diamonds to our map. To bypass obstacles, our agent needs to step onto the slab stone and jump over the acacia fence. At the same time, the track is blocked by a combination of acacia fences and fence gates, so our agent needs to discern the fence gate and open it to pass this blocking line. Meanwhile, to imporve the performance of the agent, we customized the PPO trainer with PyTorch CNN model, and optmized our reward function. Compared to the status report, out agent survive much longer and collects more diamonds. 


# Approaches


# Evaluation


# Resources Used
- [RLlib](https://docs.ray.io/en/master/rllib-training.html)
- [Pytorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Malmo API Documentation](https://microsoft.github.io/malmo/0.30.0/Documentation/index.html)
- [Malmo Tutorial](http://microsoft.github.io/malmo/0.30.0/Python_Examples/Tutorial.pdf)
- [OpenCV](https://opencv.org/)
- [Image Segmentation in OpenCV](https://realpython.com/python-opencv-color-spaces/)
- Assignment 2
