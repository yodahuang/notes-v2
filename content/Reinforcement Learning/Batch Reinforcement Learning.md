Source: [DeepMind x UCL RL Lecture Series - Function Approximation [7/13]](https://www.youtube.com/watch?v=ook46h2Jfb4&list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm&index=7)

Finding the best fitting value function given a set of paste experience instead of doing it sample by sample. 

Say if we parameterize the value function with linear functions, then an obvious choice is least squares: 

$$\begin{aligned} &\mathbb{E}[(R_{t+1} + \gamma v_{\mathbf{w}}(S_{t+1}) - v_{\mathbf{w}}(S_t))\mathbf{x}_t] = \mathbf{0} \\ \implies &\mathbf{w}_{\text{TD}} = \mathbb{E}[\mathbf{x}_t(\mathbf{x}_t - \gamma \mathbf{x}_{t+1})^{\top}]^{-1}\mathbb{E}[R_{t+1}\mathbf{x}_t] \end{aligned}$$

We can replace that expectation with sample-based approach:

$$\begin{aligned} &\frac{1}{t} \sum_{i=0}^{t} (R_{i+1} + \gamma v_{\mathbf{w}}(S_{i+1}) - v_{\mathbf{w}}(S_i))\mathbf{x}_i = \mathbf{0} \\ \implies &\mathbf{w}_{\text{LSTD}} = \left( \sum_{i=0}^{t} \mathbf{x}_i(\mathbf{x}_i - \gamma \mathbf{x}_{i+1})^{\top} \right)^{-1} \left( \sum_{i=0}^{t} R_{i+1}\mathbf{x}_i \right) \end{aligned}$$

We can update this on the fly for every new sample that comes in. That can be $O(n^3)$. But you see the inverse you would know that we can use [[Sherman-Morrison-Woodbury formula]] again:

$$\begin{aligned} \mathbf{A}_{t+1}^{-1} &= \mathbf{A}_t^{-1} - \frac{\mathbf{A}_t^{-1}\mathbf{x}_t(\mathbf{x}_t - \gamma\mathbf{x}_{t+1})^{\top} \mathbf{A}_t^{-1}}{1 + (\mathbf{x}_t - \gamma\mathbf{x}_{t+1})^{\top} \mathbf{A}_t^{-1}\mathbf{x}_t} \\ \mathbf{b}_{t+1} &= \mathbf{b}_t + R_{t+1}\mathbf{x}_t \end{aligned}$$

That leads to *Experience Replay*, which is basically staying maintaining a pool of trajectory:

Given experience consisting of trajectories of experience:

$$\mathcal{D} = \{S_0, A_0, R_1, S_1, \dots, S_t\}$$

Repeat:

1. Sample transition(s), e.g., $(S_n, A_n, R_{n+1}, S_{n+1})$ for $n \leq t$
2. Apply stochastic gradient descent update:

    $$\Delta\mathbf{w} = \alpha(R_{n+1} + \gamma v_{\mathbf{w}}(S_{n+1}) - v_{\mathbf{w}}(S_n))\nabla_{\mathbf{w}}v_{\mathbf{w}}(S_n)$$

3. Can re-use old data