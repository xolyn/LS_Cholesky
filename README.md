# LS_Cholesky
Least square problem using Cholesky Decomposition implemented by Python 

## Background and Goal
For a least square problem, we need to create an affine function, to fit data points 
$\\{(\mathbf{x} \_{i},y_i)\\}_{i=1}^n$, where $\mathbf{x}_i \in \mathbb{R}^d$ are the predictor variables of the $i$-th sample and $y_i \in \mathbb{R}$ is the corresponding response from the affine function, using matrix representation, we can construct:

$$
\mathbf{y} = 
\begin{pmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{pmatrix},
\quad\quad
\tilde{\mathbf{X}} = 
\begin{pmatrix}
-\mathbf{x}_1^T- \\
-\mathbf{x}_2^T-  \\
\vdots \\
-\mathbf{x}_n^T- 
\end{pmatrix},
\forall \mathbf{x}_i \in \mathbb{R}^{d}
$$

We know that our form for the linear regression is in the form:

$$\hat{\mathbf{y}}=\beta_0+\beta_1 \mathbf{x}\_{i1}+\beta_2 \mathbf{x}\_{i2}+\cdots +\beta_d \mathbf{x}_{id}, \forall i\quad\quad (1)$$

But observe that the $\beta$ is add to prediction for every $\mathbf{x}_i$
So instead of writing the prediction values as

$$
\beta_0 \mathbf{1} +\mathbf{X} \tilde{\boldsymbol{\beta}},
$$

where  of the appropriate size, and

$$
\mathbf{1}=
\begin{pmatrix}
    1\\
    \vdots\\
    1\\
\end{pmatrix}
\in \mathbb{R}^d,
\quad
\quad
\tilde{\boldsymbol{\beta}} = 
\begin{pmatrix}
\beta_1 \\
\vdots \\
\beta_d
\end{pmatrix}
\in \mathbb{R}^d \text{ ,}
$$

we can stack these 2 above together as:

$$
\mathbf{X} =
\begin{pmatrix}
1 & \mathbf{x}_1^T \\
1 & \mathbf{x}_2^T \\
\vdots & \vdots \\
1 & \mathbf{x}_n^T
\end{pmatrix}
\quad,\quad
\boldsymbol{\beta} = 
\begin{pmatrix}
\beta_0 \\
\beta_1 \\
\vdots \\
\beta_d
\end{pmatrix}
\in \mathbb{R}^{d+1}
$$

Then it is much easier to see that our goal is to get:

$$
\min_{\boldsymbol{\beta} \in \mathbb{R}^{d+1} } ||\mathbf{y}-\mathbf{X}\boldsymbol{\beta}||^2
\quad\quad(2),$$

which is just to find the projection of the sample point $\mathbf{y}$ onto the plane spanned by the column space of $\mathbf{X}$, using the projection function, the task above is just to find $\boldsymbol{\beta}^*$ such that:

$$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}^*=\mathbf{X}^T\mathbf{y}$$

Recall from what we learned in the lecture that we can use many decompositions for a symmetric matrix $\mathbf{X}^T\mathbf{X}$, here I will choose the easiest one for further implementation, which is the [Cholesky Decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition)

$$
\mathbf{X}^T\mathbf{X}=\mathbf{L}\mathbf{L}^T
$$
