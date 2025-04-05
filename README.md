# Mirror Descent Safe Policy Optimization (MDSPO)

MDSPO is a simple first-order method designed for **safe reinforcement learning**. It leverages **mirror descent** to maximize returns while satisfying cost constraints. The algorithm consists of three key phases:
1. **Gradient descent** (without cost constraints)
2. **Projection onto the nonparametric policy space** (with cost constraints)
3. **Projection onto the parametric policy space**

More details and experimental results will be available once this work is accepted.