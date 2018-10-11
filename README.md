# Dynamic-Programming

Value and Policy iteration algorithms applied on a 2 disk Tower of Hanoi problem.

• There are 2 disks and 3 pins. Disk A is bigger than disk B.
• The goal is to carry the disks from the first rod to the third with the order being such
that the bigger disk A is at the bottom.
• The reward for reaching the goal state is 100.
• The reward for playing the bigger disk on the smaller disk is -10.
• The reward for any other action is -1.
• When taking an action there is a 0.1 probability that the action will fail and the target
pin will be the wrong one.
• The discount factor gamma is 0.9.
