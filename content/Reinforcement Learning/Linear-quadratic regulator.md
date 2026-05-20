---
aliases:
  - LQR
date: 2026-02-21
---
Classic optimal control, state transition is linear, cost is quadratic:

$$
\min_{\mathbf{u}_1, \dots, \mathbf{u}_T} c(\mathbf{x}_1, \mathbf{u}_1) + c(f(\mathbf{x}_1, \mathbf{u}_1), \mathbf{u}_2) + \dots + c(f(f(\dots)\dots), \mathbf{u}_T)
$$

---

$$
\underbrace{f(\mathbf{x}_t, \mathbf{u}_t) = \mathbf{F}_t \begin{bmatrix} \mathbf{x}_t \\ \mathbf{u}_t \end{bmatrix} + \mathbf{f}_t}_{\text{linear}} \qquad \qquad \underbrace{c(\mathbf{x}_t, \mathbf{u}_t) = \frac{1}{2} \begin{bmatrix} \mathbf{x}_t \\ \mathbf{u}_t \end{bmatrix}^T \mathbf{C}_t \begin{bmatrix} \mathbf{x}_t \\ \mathbf{u}_t \end{bmatrix} + \begin{bmatrix} \mathbf{x}_t \\ \mathbf{u}_t \end{bmatrix}^T \mathbf{c}_t}_{\text{quadratic}}
$$

The $c$ needs to be quadratic instead of linear, or the whole thing will be a huge linear, and solving $min$ can be infinite.

LQR is a clever way to use dynamic programming so it can be solved fast.

It goes like follows:
1. Start from the last action $u_T$, suppose $x_T$ is known, what's the best action there? So we just  solve $\nabla_{u_T}Q(x_{T}, u_{T}) = \nabla_{u_T}c(x_{T}, u_{T})  = 0$. It turns out $u_{T}=K_{T}x_{T}+k_{T}$. 
2. Now we can just substitute $u_T$ with $x_T$, and it turns out... it's still linear quadratic: $V(x_{T})=\text{const} + \frac{1}{2}x^T_{T}V_Tx_{T}+x^T_Tv_T$. We are overloading $v$ a bit but you know the meaning (cost that depend only on state).
3. Now taking a step back at $T-1$. Consider $Q(x_{T-1}, u_{T-1}) = c(x_{T-1}, u_{T-1}) + V(f(x_{T-1}, u_{T-1}))$. We know from state transition that $x_{T}= f(x_{T-1}, u_{T-1})$. So now we can replace $x_T$ with $x_{T-1}$ and $u_{T-1}$ and use that $V(x_T)$ formula. Or, we propagate the "optimal value back". It's always linear quadratic.

```pseudo
\begin{algorithm}
\caption{Linear Quadratic Regulator (LQR)}
\begin{algorithmic}
\STATE \textbf{Backward recursion:}
\FOR{$t = T$ \TO $1$}
    \STATE $\mathbf{Q}_t = \mathbf{C}_t + \mathbf{F}_t^T \mathbf{V}_{t+1} \mathbf{F}_t$
    \STATE $\mathbf{q}_t = \mathbf{c}_t + \mathbf{F}_t^T \mathbf{V}_{t+1} \mathbf{f}_t + \mathbf{F}_t^T \mathbf{v}_{t+1}$
    \STATE $Q(\mathbf{x}_t, \mathbf{u}_t) = \text{const} + \frac{1}{2} \begin{bmatrix} \mathbf{x}_t \\ \mathbf{u}_t \end{bmatrix}^T \mathbf{Q}_t \begin{bmatrix} \mathbf{x}_t \\ \mathbf{u}_t \end{bmatrix} + \begin{bmatrix} \mathbf{x}_t \\ \mathbf{u}_t \end{bmatrix}^T \mathbf{q}_t$
    \STATE $\mathbf{u}_t \leftarrow \arg \min_{\mathbf{u}_t} Q(\mathbf{x}_t, \mathbf{u}_t) = \mathbf{K}_t \mathbf{x}_t + \mathbf{k}_t$
    \STATE $\mathbf{K}_t = -\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}^{-1} \mathbf{Q}_{\mathbf{u}_t, \mathbf{x}_t}$
    \STATE $\mathbf{k}_t = -\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}^{-1} \mathbf{q}_{\mathbf{u}_t}$
    \STATE $\mathbf{V}_t = \mathbf{Q}_{\mathbf{x}_t, \mathbf{x}_t} + \mathbf{Q}_{\mathbf{x}_t, \mathbf{u}_t} \mathbf{K}_t + \mathbf{K}_t^T \mathbf{Q}_{\mathbf{u}_t, \mathbf{x}_t} + \mathbf{K}_t^T \mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t} \mathbf{K}_t$
    \STATE $\mathbf{v}_t = \mathbf{q}_{\mathbf{x}_t} + \mathbf{Q}_{\mathbf{x}_t, \mathbf{u}_t} \mathbf{k}_t + \mathbf{K}_t^T \mathbf{q}_{\mathbf{u}_t} + \mathbf{K}_t^T \mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t} \mathbf{k}_t$
    \STATE $V(\mathbf{x}_t) = \text{const} + \frac{1}{2} \mathbf{x}_t^T \mathbf{V}_t \mathbf{x}_t + \mathbf{x}_t^T \mathbf{v}_t$
\ENDFOR

\STATE \textbf{Forward recursion:}
\FOR{$t = 1$ \TO $T$}
    \STATE $\mathbf{u}_t = \mathbf{K}_t \mathbf{x}_t + \mathbf{k}_t$
    \STATE $\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t)$
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

This can handle stochastic dynamics with no change of algorithm, since Gaussian is special. 

$$
\begin{aligned} f(\mathbf{x}_t, \mathbf{u}_t) &= \mathbf{F}_t \begin{bmatrix} \mathbf{x}_t \\ \mathbf{u}_t \end{bmatrix} + \mathbf{f}_t \\ \mathbf{x}_{t+1} &\sim p(\mathbf{x}_{t+1}|\mathbf{x}_t, \mathbf{u}_t) \\ p(\mathbf{x}_{t+1}|\mathbf{x}_t, \mathbf{u}_t) &= \mathcal{N}\left(\mathbf{F}_t \begin{bmatrix} \mathbf{x}_t \\ \mathbf{u}_t \end{bmatrix} + \mathbf{f}_t, \Sigma_t\right) \end{aligned}
$$

## The nonlinear case: iLQR / ddP

The obvious idea is: just like how [[Extended Kalman Filter|EKF]] expends KF, we use Taylor expansion here:

$$
\begin{aligned} f(\mathbf{x}_t, \mathbf{u}_t) &\approx f(\hat{\mathbf{x}}_t, \hat{\mathbf{u}}_t) + \nabla_{\mathbf{x}_t, \mathbf{u}_t} f(\hat{\mathbf{x}}_t, \hat{\mathbf{u}}_t) \begin{bmatrix} \mathbf{x}_t - \hat{\mathbf{x}}_t \\ \mathbf{u}_t - \hat{\mathbf{u}}_t \end{bmatrix} \\ c(\mathbf{x}_t, \mathbf{u}_t) &\approx c(\hat{\mathbf{x}}_t, \hat{\mathbf{u}}_t) + \nabla_{\mathbf{x}_t, \mathbf{u}_t} c(\hat{\mathbf{x}}_t, \hat{\mathbf{u}}_t) \begin{bmatrix} \mathbf{x}_t - \hat{\mathbf{x}}_t \\ \mathbf{u}_t - \hat{\mathbf{u}}_t \end{bmatrix} + \frac{1}{2} \begin{bmatrix} \mathbf{x}_t - \hat{\mathbf{x}}_t \\ \mathbf{u}_t - \hat{\mathbf{u}}_t \end{bmatrix}^T \nabla_{\mathbf{x}_t, \mathbf{u}_t}^2 c(\hat{\mathbf{x}}_t, \hat{\mathbf{u}}_t) \begin{bmatrix} \mathbf{x}_t - \hat{\mathbf{x}}_t \\ \mathbf{u}_t - \hat{\mathbf{u}}_t \end{bmatrix} \end{aligned}
$$

Now you can see the $x_{t}- \hat{x}_t$ and $u_{t}- \hat{u}_t$ part is just like the previous case and we can run LQR on the $\delta$ term.
```pseudo
\begin{algorithm}
\caption{Iterative LQR (iLQR)}
\begin{algorithmic}
\REPEAT
    \FOR{$t = 1$ \TO $T$}
        \STATE $\mathbf{F}_t = \nabla_{\mathbf{x}_t, \mathbf{u}_t} f(\hat{\mathbf{x}}_t, \hat{\mathbf{u}}_t)$ \COMMENT{Linearize dynamics}
        \STATE $\mathbf{c}_t = \nabla_{\mathbf{x}_t, \mathbf{u}_t} c(\hat{\mathbf{x}}_t, \hat{\mathbf{u}}_t)$ \COMMENT{Quadratic cost approximation (gradient)}
        \STATE $\mathbf{C}_t = \nabla_{\mathbf{x}_t, \mathbf{u}_t}^2 c(\hat{\mathbf{x}}_t, \hat{\mathbf{u}}_t)$ \COMMENT{Quadratic cost approximation (Hessian)}
    \ENDFOR
    
    \STATE Run LQR backward pass on state $\delta \mathbf{x}_t = \mathbf{x}_t - \hat{\mathbf{x}}_t$ and action $\delta \mathbf{u}_t = \mathbf{u}_t - \hat{\mathbf{u}}_t$
    
    \STATE Run forward pass with real nonlinear dynamics and $\mathbf{u}_t = \mathbf{K}_t(\mathbf{x}_t - \hat{\mathbf{x}}_t) + \mathbf{k}_t + \hat{\mathbf{u}}_t$
    
    \STATE Update $\hat{\mathbf{x}}_t$ and $\hat{\mathbf{u}}_t$ based on states and actions in forward pass
\UNTIL{convergence}
\end{algorithmic}
\end{algorithm}
```
Note this can be bad since the Newton's method overshoots. we can add an $\alpha$ to $k_t$ part in forward pass to see if we see improvement / line search. 
This is the same idea of [[Newton's method]], it's an approximation of [[Newton's method]] for solving the min.
```pseudo
\begin{algorithm}
\caption{Newton's Method for Optimization}
\begin{algorithmic}
\REPEAT
    \STATE $\mathbf{g} = \nabla_{\mathbf{x}} g(\hat{\mathbf{x}})$ \COMMENT{Compute gradient}
    \STATE $\mathbf{H} = \nabla_{\mathbf{x}}^2 g(\hat{\mathbf{x}})$ \COMMENT{Compute Hessian}
    \STATE $\hat{\mathbf{x}} \leftarrow \arg \min_{\mathbf{x}} \frac{1}{2}(\mathbf{x} - \hat{\mathbf{x}})^T \mathbf{H} (\mathbf{x} - \hat{\mathbf{x}}) + \mathbf{g}^T (\mathbf{x} - \hat{\mathbf{x}})$
\UNTIL{convergence}
\end{algorithmic}
\end{algorithm}
```
If you use second order dynamics for $f$ too, that becomes DDP.