Here I put some useful links that helped me understanding "why Gaussian / normal" distribution is such a special distribution.

It's commonly used in Robotics not because of its expressiveness, but because it can be reasoned in close form, and can be parameterized by its moments, which is very convenient.

Why it can be reasoned in close form? Say $A$ and $B$ are random variables of normal distribution, then `A + B` also forms normal distribution. That's where we got the central limit theorem, adding stuff up and we always got Gaussian.

Heres's 3Blue1Brown series:
- [Convolutions and adding random variables](https://www.3blue1brown.com/lessons/convolutions2)
- [A pretty reason why Gaussian + Gaussian = Gaussian](https://www.3blue1brown.com/lessons/gaussian-convolution)
The core idea there (I think) $\int e^{x}= e^x$ is too convenient. 

## Moment parameterization
$$P(x)\propto\exp\left(-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$$
## Natural parameterization
Also known as canonical parameterization. 
$$
P(x) \propto \exp \left(J^{T}x - \frac{1}{2}x^{T}Px\right)
$$
$$
\begin{aligned}
P&=\Sigma^{-1} \\
J&=\Sigma^{-1}\mu
\end{aligned}$$
$J$: information vector, $P$: matrix.