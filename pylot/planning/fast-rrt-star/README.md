# Fast RRT*

Here a fast version of RRT* is implemented. Due to design choices and modifications made to original algorithm, performance was found to be around 10x faster (Check the metrics plots).

### Installation

* Install Python 3
* Install pip
* run $> pip install -r requirements.txt

### Running the Code
Code for RRT and fast RRT* is in 2 different files. Call the method directly passing the parameters. Algorithm returns a tree and a final state. To get the path, run shortest path finding algorithm in the tree using initial and final states. A sample file is provided in Main.py.

## Space Representation

The code accepts representation of a space in the form ((origin),(range)). For e.g. if an Obstacle's origin is at (15,10) and size is (5,5), then its space will be represented as ((15,10),(5,5)). Algorithm needs a state space, a starting state, a target region and an obstacle map containing information about all the obstacles.

### Parameters
##### state_space
State space is the working world for the algorithm. It is passed in the form defined above.
##### starting_state
Starting state the state in which agent starts. It is passed as a single origin point.
##### target_space
Target space is the endgame region for the algorithm. It is passed in the form defined above.
##### obstacle_map
Obstacle map contains information every obstacle in the state space. Each obstacle space is defined in the form discussed above.
##### n_samples
Number of iterations for which the algorithm will sample points. Collided or out of space points are considered as an iteration so there may be lesser samples returned than this value. Default value is 1000.
##### granularity
As incremental collision checking technique is used, so granularity for collision check is required. Finer the granularity, slower the program will be.
##### d_threshold
It defines how far the new point should be sampled from the existing node.

### Authors

* **Dixant Mittal** - [dixantmittal](https://github.com/dixantmittal)

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details
