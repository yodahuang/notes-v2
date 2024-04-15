---
aliases:
  - quantile loss
---
A loss useful for [quantile regression](https://en.wikipedia.org/wiki/Quantile_regression). Penalize undershoot or overshoot one side than the other side.
$$
L_{\tau}(y, f(x)) = \begin{cases} \tau \cdot |y - f(x)| & \text{if } y \geq f(x) \\ (1 - \tau) \cdot |y - f(x)| & \text{if } y < f(x) \end{cases}
$$
So it may look like this: ![[pinball_loss.png]]

> For 𝜏=0.1, we would expect the model to underpredict 90% of the time and overpredict 10% of the time. And now we have already bridged the gap to distributions: This is equivalent to the 10% quantile.

Reference:
- This blog: [How I made peace with quantile regression](https://mindfulmodeler.substack.com/p/how-i-made-peace-with-quantile-regression)