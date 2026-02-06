---
date: 2025-09-14
---


This note is a summary of [Master Positional Encoding: Part I](https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3)

$$
\begin{aligned}PE_{(pos,2i)}=sin(pos/10000^{2i/d_{\mathrm{model}}})\\PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{\mathrm{model}}})\end{aligned}
$$

Think about the position encoder first as our normal binary encoding: 000, 001, 010, 011, etc. That's not continuous. It cannot be interpolated, and has jump. So that's why there is `sin` there. 
![[positional_encoding.png]]
Why do we have both cosine and sin though? That's to make sure there's a nice property:

$$
\mathrm{PE}(x+\Delta x)=\mathrm{PE}(x)\cdot\mathbf{T}(\Delta x)
$$

and with both `sin` and `cos`, we have this:

$$
\left.\left(\begin{array}{c}\cos(\theta+\phi)\\\sin(\theta+\phi)\end{array}\right.\right)=\begin{pmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{pmatrix}\begin{pmatrix}\cos\theta\\\sin\theta\end{pmatrix}
$$

and we can find such a $T(\Delta x)$ as 

$$
\begin{aligned}
\mathbf{v}^{(i)}&=[\cos(\omega_0x_i),\sin(\omega_0x_i),\ldots,\cos(\omega_{n-1}x_i),\sin(\omega_{n-1}x_i)]\\
T(\Delta x) &= 
\left( 
  \begin{array}{ccc}
    \begin{bmatrix}
      \cos(\omega_0\Delta x) & -\sin(\omega_0\Delta x) \\
      \sin(\omega_0\Delta x) & \cos(\omega_0\Delta x)
    \end{bmatrix} 
    & \cdots & 0 \\
    \vdots & \ddots & \vdots \\
    0 & \cdots & 
    \begin{bmatrix}
      \cos(\omega_{n-1}\Delta x) & -\sin(\omega_{n-1}\Delta x) \\
      \sin(\omega_{n-1}\Delta x) & \cos(\omega_{n-1}\Delta x)
    \end{bmatrix}
  \end{array} 
\right)
\end{aligned}
$$