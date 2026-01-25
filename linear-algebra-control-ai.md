# Linear Algebra for Control Theory and AI: A Comprehensive Reference

This document is a personal technical reference for linear algebra concepts used in control theory, estimation, and modern AI methods.

The purpose is not to re-teach linear algebra from first principles, nor to provide formal mathematical proofs. Instead, the goal is to preserve **working understanding** — the definitions, identities, and derivations that repeatedly appear when reasoning about:

* state-space models
* system modes and eigenvalues
* controllability and observability
* estimation and covariance propagation
* optimization and least-squares problems
* dimensionality reduction and learning algorithms

This document exists so that future-me can re-enter the mathematics quickly, reconstruct reasoning steps, and reconnect equations to physical and algorithmic meaning.

Where appropriate, intermediate steps are shown explicitly, even when they may appear redundant, because the objective is **recoverability of understanding**, not conciseness.

---

## Document Structure

The material is organized into three parts:

* **Part 1 — Fundamentals**  
  Core linear algebra concepts required for all later sections.

* **Part 2 — Control Theory Applications**  
  How linear algebra appears in state-space modeling, stability, estimation, and optimal control.

* **Part 3 — AI and Machine Learning Applications**  
  Linear algebra structures that appear in data analysis, learning algorithms, and modern numerical methods.

Each section is written to stand alone as a reference.

---

## Part 1: Fundamentals of Linear Algebra

This section establishes the basic mathematical objects and operations that appear repeatedly in control systems and estimation theory.

The emphasis is on **structure and interpretation**, not abstraction for its own sake.

---

### 1. Vectors and Matrices

#### Definitions

Vectors and matrices are the fundamental objects used to represent system variables and linear relationships.

In control theory, vectors most commonly represent:

* system state
* inputs and disturbances
* outputs and measurements

Matrices represent how these quantities interact.

A vector in $\mathbb{R}^n$ is written as:



$$
\mathbf{v} = \begin{bmatrix}v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
$$


A matrix with $m$ rows and $n$ columns is written as:



$$
\mathbf{A} = \begin{bmatrix}a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}\end{bmatrix}
$$


Matrices act on vectors to produce new vectors, forming the basis of linear system modeling.

---

### Operations

#### Vector Addition

For vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:



$$
\mathbf{u} + \mathbf{v} = \begin{bmatrix}u_1 + v_1 \\ u_2 + v_2 \\
\vdots \\ u_n + v_n\end{bmatrix}
$$


Vector addition corresponds to component-wise accumulation.

This operation appears frequently when combining state contributions, disturbances, or modal responses.

---

#### Scalar Multiplication

For a scalar $c \in \mathbb{R}$:



$$
c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n\end{bmatrix}
$$


Scalar multiplication represents uniform scaling of a vector and is fundamental to linear superposition.

---

#### Matrix Addition

For matrices of identical dimensions, addition is performed element-wise.

This operation typically arises when combining system dynamics from multiple effects (e.g., nominal model plus perturbation).

---

#### Matrix Multiplication

Matrix multiplication defines how linear transformations compose.

Let:



$$
\mathbf{A} \in \mathbb{R}^{m \times p}, \quad \mathbf{B} \in \mathbb{R}^{p \times n}
$$


Then:



$$
\mathbf{C} = \mathbf{A}\mathbf{B}
$$


with elements:



$$
c_{ij} = \sum_{k=1}^{p} a_{ik} b_{kj}
$$


This operation corresponds to applying transformation $B$ first, followed by transformation $A$.

In control systems, matrix multiplication encodes:

* state propagation
* coordinate transformations
* coupling between subsystems

---

#### Worked Example

The step-by-step arithmetic is intentionally preserved so that the mechanics of multiplication can be reconstructed without mental shortcuts.

##### Step 1: Define the matrices

Let



$$
A =\begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix},\quad
B =\begin{bmatrix}9 & 8 & 7 \\ 6 & 5 & 4 \\ 3 & 2 & 1\end{bmatrix}
$$


---

##### Step 2: Recall the multiplication rule

If $C = A \times B$, then each element $c_{ij}$ in $C$ is calculated as:



$$
c_{ij} = \sum_{k=1}^{3} a_{ik} \cdot b_{kj}
$$


That is, **row $i$ of $A$ dot column $j$ of $B$.**

---

##### Step 3: Compute each element

###### First row of $C$



$$
c_{11} = (1)(9) + (2)(6) + (3)(3) = 9 + 12 + 9 = 30
$$




$$
c_{12} = (1)(8) + (2)(5) + (3)(2) = 8 + 10 + 6 = 24
$$




$$
c_{13} = (1)(7) + (2)(4) + (3)(1) = 7 + 8 + 3 = 18
$$


So first row is: $[30 \quad 24 \quad 18]$.

---

###### Second row of $C$



$$
c_{21} = (4)(9) + (5)(6) + (6)(3) = 36 + 30 + 18 = 84
$$




$$
c_{22} = (4)(8) + (5)(5) + (6)(2) = 32 + 25 + 12 = 69
$$




$$
c_{23} = (4)(7) + (5)(4) + (6)(1) = 28 + 20 + 6 = 54
$$


So second row is: $[84 \quad 69 \quad 54]$.

---

###### Third row of $C$



$$
c_{31} = (7)(9) + (8)(6) + (9)(3) = 63 + 48 + 27 = 138
$$




$$
c_{32} = (7)(8) + (8)(5) + (9)(2) = 56 + 40 + 18 = 114
$$




$$
c_{33} = (7)(7) + (8)(4) + (9)(1) = 49 + 32 + 9 = 90
$$


So third row is: $[138 \quad 114 \quad 90]$.

---

##### Step 4: Final result



$$
C = A \times B =\begin{bmatrix}30 & 24 & 18 \\ 84 & 69 & 54 \\ 138 & 114 & 90\end{bmatrix}
$$


---

### Transpose and Special Matrices

The transpose operation exchanges rows and columns:



$$
(\mathbf{A}^T)_{ij} = a_{ji}
$$


Several special matrix types appear repeatedly in system analysis:

* **Identity matrix** $\mathbf{I}$: leaves vectors unchanged
* **Diagonal matrices**: represent decoupled systems
* **Symmetric matrices**: arise in energy and covariance expressions
* **Orthogonal matrices**: preserve length and angle

Orthogonal matrices are especially important in coordinate transformations, modal analysis, and numerical stability.

---

### 2. Systems of Linear Equations

A system of linear equations is written compactly as:



$$
\mathbf{A}\mathbf{x} = \mathbf{b}
$$


This form appears throughout control theory when solving for:

* equilibrium points
* steady-state responses
* parameter estimates
* least-squares solutions

#### Gaussian Elimination
Gaussian elimination transforms a system into row-echelon form using elementary row operations.
The objective is not conceptual insight, but reliable solution extraction.
A system of linear equations can be written as $\mathbf{A}\mathbf{x} = \mathbf{b}$, where $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$, and $\mathbf{b} \in \mathbb{R}^m$.
- **Gaussian Elimination**: Transform $\mathbf{A}$ into row echelon form using elementary row operations, then solve via back-substitution.
- **Example**: Solve 
```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 5 \\ 11 \end{bmatrix}
```
- Row reduce to 
```math
\begin{bmatrix} 1 & 2 \\ 0 & -2 \end{bmatrix}\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 5 \\ -4 \end{bmatrix}
```
- Solution: $x_2 = 2$, $x_1 = 1$.
---

### Rank

Matrix rank determines the number of independent equations or directions represented.

Rank conditions directly govern:

* existence of solutions
* uniqueness of solutions
* controllability and observability

- **Rank**: The rank of $\mathbf{A}$ is the maximum number of linearly independent rows or columns.
  - If $\text{rank}(\mathbf{A}) = \text{rank}([\mathbf{A}\mid \mathbf{b}])$:
    - If $\text{rank}(\mathbf{A}) = n$: Unique solution.
    - If $\text{rank}(\mathbf{A}) < n$: Infinitely many solutions.
  - If $\text{rank}(\mathbf{A}) < \text{rank}([\mathbf{A}\mid \mathbf{b}])$: No solution.

---

### 3. Matrix Properties

#### Determinant

The determinant provides information about:

* invertibility
* volume scaling
* singularity

A zero determinant indicates loss of dimensionality and non-invertibility.

---

#### Inverse

If $\det(\mathbf{A}) \neq 0$, an inverse exists such that:



$$
\mathbf{A}^{-1}\mathbf{A} = \mathbf{I}
$$


In practice, explicit inversion is often avoided numerically, but the concept remains central for theoretical reasoning.

---

#### Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors describe intrinsic system behavior.

For a matrix $\mathbf{A}$:



$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}
$$


This equation identifies directions that remain aligned under the transformation $\mathbf{A}$.

In control theory, eigenvalues correspond directly to system modes and stability characteristics.

- For an LTI system, stability depends on the eigenvalues of $\mathbf{A}$:
  - **Asymptotically stable**: All eigenvalues have negative real parts.
  - **Unstable**: At least one eigenvalue has a positive real part.
  - **Example**: $\mathbf{A} = \begin{bmatrix} -1 & 0 \\ 0 & -2 \end{bmatrix}$, eigenvalues $\lambda = -1, -2$, stable.

---

#### Diagonalization

If a full set of independent eigenvectors exists:



$$
\mathbf{A} = \mathbf{P}\mathbf{D}\mathbf{P}^{-1}
$$


Diagonalization separates coupled dynamics into independent modal components.

This concept underlies modal analysis, decoupling, and state transformations.

---

### 4. Orthogonality

Orthogonality formalizes the idea of independence under inner products.

It appears naturally in:

* coordinate systems
* projections
* estimation theory
* least-squares problems

Dot products, norms, and projections should be treated as geometric tools, not just algebraic operations.

- **Dot Product**: For $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:



$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i
$$


- **Orthogonal Vectors**: If $\mathbf{u} \cdot \mathbf{v} = 0$.
- **Norm**: $\|\mathbf{v}\| = \sqrt{\mathbf{v} \cdot \mathbf{v}}$.
- **Orthogonal Projection**: Projection of $\mathbf{v}$ onto $\mathbf{u}$:



$$
\text{proj}_{\mathbf{u}} \mathbf{v} = \left( \frac{\mathbf{u} \cdot \mathbf{v}}{\mathbf{u} \cdot \mathbf{u}} \right)\mathbf{u}
$$


- **Gram-Schmidt Process**: Orthogonalize a set of vectors:
  - Given $\{\mathbf{v}_1, \mathbf{v}_2\}$, set $\mathbf{u}_1 = \mathbf{v}_1$, $\mathbf{u}_2 = \mathbf{v}_2 - \text{proj}_{\mathbf{u}_1} \mathbf{v}_2$.
- **Orthogonal Matrix**: $\mathbf{Q}^T \mathbf{Q} = \mathbf{I}$, preserves dot products.

---

### 5. Vector Spaces

Vector spaces provide the framework in which all previous objects live.

Understanding span, basis, and dimension is essential for reasoning about:

* reachable subspaces
* observable subspaces
* reduced-order models

Linear transformations map between vector spaces and are represented concretely by matrices.

---

# Part 2: Advanced Linear Algebra for Control Theory

This section focuses on how linear algebra appears in dynamic system modeling, estimation, and feedback control.

Unlike Part 1, where the emphasis is on mathematical structure, the emphasis here is on **interpretation** — how matrix properties translate directly into system behavior.

---

## 1. State-Space Representation

A continuous-time linear time-invariant (LTI) system can be written as:



$$
\dot{\mathbf{x}}(t) = \mathbf{A}\mathbf{x}(t) + \mathbf{B}\mathbf{u}(t)
$$




$$
\mathbf{y}(t) = \mathbf{C}\mathbf{x}(t) + \mathbf{D}\mathbf{u}(t)
$$


where:

* $\mathbf{x}(t) \in \mathbb{R}^n$ is the state vector
* $\mathbf{u}(t) \in \mathbb{R}^m$ is the input vector
* $\mathbf{y}(t) \in \mathbb{R}^p$ is the output vector

and:

* $\mathbf{A}$ encodes system dynamics
* $\mathbf{B}$ maps inputs into state evolution
* $\mathbf{C}$ maps state to measured outputs
* $\mathbf{D}$ represents direct feedthrough

The matrix $\mathbf{A}$ plays a central role: it defines the system’s intrinsic behavior in the absence of input.

---

### State Evolution

The general solution is:



$$
\mathbf{x}(t) = e^{\mathbf{A}t}\mathbf{x}(0) +\int_0^t e^{\mathbf{A}(t-\tau)}\mathbf{B}\mathbf{u}(\tau)\, d\tau
$$


This expression separates system behavior into:

* **natural response** governed solely by $\mathbf{A}$
* **forced response** driven by inputs through $\mathbf{B}$

The matrix exponential is therefore not merely a mathematical construct — it directly describes how system modes evolve over time.

---

## 2. Controllability and Observability

Controllability and observability are fundamentally **rank properties of matrices**, not abstract system notions.

Linear algebra provides the precise criteria.

---

### Controllability

A system is controllable if it is possible to drive the state from any initial condition to any desired final condition using admissible inputs.

The controllability matrix is:



$$
\mathbf{C} = \begin{bmatrix}\mathbf{B} & \mathbf{A}\mathbf{B} &
\cdots & \mathbf{A}^{n-1}\mathbf{B}\end{bmatrix}
$$


If this matrix has full row rank, the system is controllable.

Each successive term represents how input influence propagates through system dynamics over time.

Loss of rank implies that certain state directions cannot be excited, regardless of input.

---

### Observability

A system is observable if its internal state can be reconstructed from output measurements.

The observability matrix is:



$$
\mathbf{O} =\begin{bmatrix}\mathbf{C} \\ \mathbf{C}\mathbf{A} \\ \vdots \\ \mathbf{C}\mathbf{A}^{n-1}\end{bmatrix}
$$


If this matrix has full column rank, the system is observable.

Each row block corresponds to observing how internal dynamics project onto the output over time.

Loss of observability implies that certain internal modes exist but cannot be detected through measurement.

---

## 3. Stability Analysis

For continuous-time LTI systems, stability is determined entirely by the eigenvalues of the system matrix $\mathbf{A}$.

* All eigenvalues with negative real parts → asymptotically stable
* At least one eigenvalue with positive real part → unstable
* Eigenvalues on the imaginary axis → marginal behavior

Eigenvalues therefore represent **natural system modes** — directions in state space that evolve independently under the dynamics.

This provides the direct bridge between linear algebra and physical system behavior.

---

## 4. Linear Quadratic Regulator (LQR)

The Linear Quadratic Regulator problem formalizes optimal feedback control as a quadratic optimization problem.

The cost function is:



$$
J = \int_0^\infty\left(\mathbf{x}^T\mathbf{Q}\mathbf{x}+\mathbf{u}^T\mathbf{R}\mathbf{u}\right)dt
$$


where:

* $\mathbf{Q} \ge 0$ penalizes state deviation
* $\mathbf{R} > 0$ penalizes control effort

The optimal control law takes the form:



$$
\mathbf{u} = -\mathbf{K}\mathbf{x}
$$


with gain:



$$
\mathbf{K} = \mathbf{R}^{-1}\mathbf{B}^T\mathbf{P}
$$


The matrix $\mathbf{P}$ satisfies the continuous-time algebraic Riccati equation:



$$
\mathbf{A}^T\mathbf{P} + \mathbf{P}\mathbf{A}- \mathbf{P}\mathbf{B}\mathbf{R}^{-1}\mathbf{B}^T\mathbf{P} +\mathbf{Q}=\mathbf{0}
$$


The Riccati equation itself is a nonlinear matrix equation whose structure arises entirely from linear algebraic identities.

---

## 5. Kalman Filter

The Kalman filter provides optimal state estimation in the presence of process and measurement noise.

It relies on covariance propagation, which is purely a matrix operation.

---

### Prediction



$$
\hat{\mathbf{x}}_{k|k-1} =\mathbf{A}\hat{\mathbf{x}}_{k-1|k-1}+\mathbf{B}\mathbf{u}_{k-1}
$$




$$
\mathbf{P}_{k|k-1} = \mathbf{A}\mathbf{P}_{k-1|k-1}\mathbf{A}^T+\mathbf{Q}
$$


---

### Update



$$
\mathbf{K}_k =\mathbf{P}_{k|k-1}\mathbf{C}^T\left(\mathbf{C}\mathbf{P}_{k|k-1}\mathbf{C}^T+\mathbf{R}\right)^{-1}
$$




$$
\hat{\mathbf{x}}_{k|k} =\hat{\mathbf{x}}_{k|k-1}+\mathbf{K}_k\left(\mathbf{y}_k-\mathbf{C}\hat{\mathbf{x}}_{k|k-1}\right)
$$




$$
\mathbf{P}_{k|k} =(\mathbf{I} - \mathbf{K}_k\mathbf{C})\mathbf{P}_{k|k-1}
$$


Each step is a direct application of linear algebraic transformations.

The Kalman filter can be interpreted as continuously projecting uncertainty into and out of measurement subspaces.

---

# Part 3: Linear Algebra in AI and Machine Learning

Many mathematical tools used in modern AI are direct extensions of concepts already present in control theory and estimation.

From a linear algebra perspective, machine learning methods rely heavily on:

* subspace decomposition
* projections
* least-squares optimization
* covariance structure
* dimensionality reduction

Seen this way, AI techniques are not conceptually separate from control theory — they operate on the same mathematical objects under different problem formulations.

---

## 1. Singular Value Decomposition (SVD)

Any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ can be decomposed as:



$$
\mathbf{A}=\mathbf{U}\mathbf{\Sigma}\mathbf{V}^T
$$


where:

* $\mathbf{U}$ contains orthonormal left singular vectors
* $\mathbf{V}$ contains orthonormal right singular vectors
* $\mathbf{\Sigma}$ contains nonnegative singular values

SVD generalizes eigenvalue decomposition to non-square matrices.

It provides a basis in which the action of $\mathbf{A}$ becomes purely scaling.

This decomposition is fundamental to numerical stability and model reduction.

---

### Interpretation

Each singular value represents the gain of the matrix along a particular direction.

Large singular values correspond to dominant directions.  
Small singular values correspond to directions that are weakly observable or numerically fragile.

This interpretation closely parallels controllability and observability concepts in control systems.

---

## 2. Least Squares Approximation

Least squares problems arise when a system of equations has no exact solution.

Given:



$$
\mathbf{A}\mathbf{x} = \mathbf{b}
$$


with more equations than unknowns, the objective becomes minimizing the residual:



$$
\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2
$$


The solution is:



$$
\mathbf{x}=(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}
$$


This expression results from projecting $\mathbf{b}$ onto the column space of $\mathbf{A}$.

Least squares is the foundation of parameter estimation, regression, and system identification.

---

## 3. Positive Definite Matrices

A matrix $\mathbf{A}$ is positive definite if:



$$
\mathbf{x}^T\mathbf{A}\mathbf{x} > 0\quad \text{for all } \mathbf{x} \neq 0
$$


Such matrices possess several important properties:

* symmetry
* strictly positive eigenvalues
* invertibility

Positive definite matrices appear naturally as:

* covariance matrices
* weighting matrices in optimization
* Hessians in convex problems

Their structure guarantees well-posed quadratic forms.

---

## 4. Matrix Calculus

Matrix calculus provides tools for computing derivatives of scalar functions with respect to vectors and matrices.

A common identity is:



$$
\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T\mathbf{A}\mathbf{x})=(\mathbf{A} + \mathbf{A}^T)\mathbf{x}
$$


These identities underpin gradient-based optimization algorithms.

In both control and machine learning, gradient descent emerges from repeated linear approximations governed by matrix derivatives.

---

## 5. Principal Component Analysis (PCA)

Principal Component Analysis performs dimensionality reduction by identifying directions of maximum variance.

The covariance matrix is defined as:



$$
\mathbf{C}=\frac{1}{n-1}\mathbf{X}^T\mathbf{X}
$$


Eigen decomposition yields:



$$
\mathbf{C}=\mathbf{P}\mathbf{D}\mathbf{P}^T
$$


The eigenvectors corresponding to the largest eigenvalues define dominant data directions.

Projection onto these directions retains maximum variance while reducing dimensionality.

This procedure is mathematically equivalent to selecting dominant singular vectors.

---

## 6. Neural Networks

A neural network layer can be written compactly as:



$$
\mathbf{y}=f(\mathbf{W}\mathbf{x} + \mathbf{b})
$$


where:

* $\mathbf{W}$ is a weight matrix
* $\mathbf{b}$ is a bias vector
* $f(\cdot)$ is a nonlinear activation

Linear algebra governs how signals propagate through layers.

Nonlinearities provide expressive power, but the underlying transformations remain matrix operations.

Backpropagation relies on repeated application of matrix calculus and chain rules.

---

## 7. Convolution in CNNs

Convolution operations can be expressed as matrix multiplications through structured matrices such as Toeplitz matrices.

This formulation reveals that convolutional networks are still linear systems locally, with parameter sharing imposing structural constraints.

Understanding convolution in matrix form clarifies:

* receptive fields
* dimensionality growth
* computational complexity

---

## 8. Graph Theory in AI

Graphs are represented algebraically through matrices.

* **Adjacency matrix** encodes connectivity
* **Degree matrix** encodes node degree
* **Laplacian matrix**:



$$
\mathbf{L} = \mathbf{D} - \mathbf{A}
$$


The Laplacian plays a central role in:

* spectral clustering
* diffusion processes
* graph neural networks

Eigenstructure of the Laplacian reveals connectivity and flow properties of the graph.

---

## Conclusion

This document consolidates the linear algebra framework that underlies control theory, estimation, and modern machine learning.

The intent is not completeness, but continuity — maintaining a coherent mathematical foundation that connects:

* system dynamics
* optimization
* estimation
* learning

By preserving explicit derivations and structural interpretations, this reference is designed to support long-term technical reasoning and future re-entry into complex system analysis.

---

### Final note to future-self

If this document feels dense on re-reading, that is expected.

It is meant to serve as a **map**, not a lesson — a place to reconnect equations to meaning before diving back into detailed derivations elsewhere.
