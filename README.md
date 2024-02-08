# CollisionPro


```A framework for collision probability distribution estimation via temporal difference learning.```

---

__NOTE__ : The code will be uploaded after the final submission of our paper.

---

### Qualitative Demonstration

<p align="center">
    <img src="assets/demo.gif" alt="Demo GIF" />
</p>

The video depicts a simple environment where each object functions as a mass-spring-damper system. 
The ego circle (grey) moves with a constant linear velocity and can exert upward or downward force to evade obstacles (red circles). 
Obstacles oscillate vertically and can apply horizontal force to avoid the ego circle. 
Their behavior is unpredictable, as each obstacle has unique parameters governing evasion tactics. 
The state is represented by stacking radius, position, velocity, and acceleration over three time steps. 
The lower-left panel displays the historical collision probability within an eight-second time horizon. 
The lower-right panel exhibits the current cumulative collision probability distribution over the time horizon.
