# Black Hole Strategy in Metal-Organic Framework (MOF) Graph

<p align="center">
    <img src="BH.jpg" alt="Black Hole Strategy in Metal-Organic Framework (MOF) Graph based on MOFGalaxyNet" width="600">
</p>


# Black Hole Strategy for Node Removal

## Overview
This repository contains a Python-based implementation of the **Black Hole Strategy** applied to community detection in networks. The main goal is to identify and remove nodes from a graph based on a gravity metric, which takes into account degree centrality, betweenness centrality, and edge weights. The method allows for the removal of a percentage of nodes with the lowest gravity in each community based on a specified threshold.

## Demo!
Below is an example of the **Black Hole Strategy** in action, showing the graph as nodes are highlighted based on their gravity:


<p align="center">
    <img src="Animated_BH_txt_shorter.gif" alt="Black Hole Strategy in Metal-Organic Framework (MOF) Graph based on MOFGalaxyNet" width="600">
</p>


## How it Works

1. **Community Detection**: First, the graph is divided into communities using the **Girvan-Newman algorithm**.
2. **Gravity Calculation**: For each community, the gravity score for each node is computed using normalized degree centrality, betweenness centrality, and edge weights.
3. **Stratified Sampling**: Nodes are categorized into bins according to their PLD (Pore Limiting Diameter) values, and a proportional number of nodes are selected from each bin to maintain the PLD distribution. (PLD, being a critical parameter in MOFs data, can be substituted or complemented by other properties depending on the application.)
4. **Node Removal**: Nodes with the lowest gravity are removed based on a configurable **threshold**, representing the percentage of nodes to be removed in each community.
5. **Results**: The remaining nodes are analyzed, and the reduced graph is visualized.



## MOFGalaxyNet and Black Hole Strategy
- **MOFGalaxyNet:**
    <p>To access the related code for MOFGalaxyNet, visit the following GitHub repository:</p>
    <a href="https://github.com/MehrdadJalali-KIT/MOFGalaxyNet">MOFGalaxyNet GitHub Repository</a>



