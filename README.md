# DIRAL
Distributed Resource Allocation with Multi-Agent Deep Reinforcement Learning for 5G-V2V Communication. This repo contains the source code of the toy example that we used in our paper to test the performance of the algorithm.

## Abstract
We consider the distributed resource selection problem in Vehicle-to-vehicle (V2V) communication in the absence of a base station. Each vehicle autonomously selects transmission resources from a pool of shared resources to disseminate Cooperative Awareness Messages (CAMs). This is a consensus problem where each vehicle has to select a unique resource. The problem becomes more challenging whenódue to mobilityóthe number of vehicles in vicinity of each other is changing dynamically. In a congested scenario, allocation of unique resources for each vehicle becomes infeasible and a congested resource allocation strategy has to be developed. The standardized approach in 5G, namely semi-persistent scheduling(SPS) suffers from effects caused by spatial distribution of the vehicles. In our approach, we turn this into an advantage. We propose a novel DIstributed Resource Allocation mechanism using multi-agent reinforcement Learning (DIRAL) which builds on a unique state representation. One challenging issue is to cope with the non-stationarity introduced by concurrently learning agents which causes convergence problems in multi-agent learning systems. We aimed to tackle non-stationarity with unique state representation. Specifically, we deploy view-based positional distribution as a state representation to tackle non-stationarity and perform complex joint behavior in a distributed fashion. Our results showed that DIRAL improves PRR by %20 compared to SPS in challenging congested scenarios.

## Algorithm Architecture
* Double Deep Q Network(DQN) with Long-Term-Short-Memory(LSTM)
* Parameter sharing
* Centralized training, decentralized execution framework
<img src="/assets/training.png" width="60%"> 

## Demo
Illustration of the trained model in the real-time network simulator(RealNeS) with 6 vehicles and 5 available resources e.g. congested scenario. As we can see, far vehicles choose the same resources autonomously based on their local observations whereas near vehicles use separate resources.

<img src="/assets/diral_demo.gif" width="110%">

## Requirements
