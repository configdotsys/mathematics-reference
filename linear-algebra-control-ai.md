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
\mathbf{A} = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
$$

Matrices act on vectors to produce new vectors. Rather than repeatedly expanding matrix–vector products, this document adopts the compact symbolic form

$$
\mathbf{y} = \mathbf{A}\mathbf{x}
$$

which will be used consistently throughout.

---

### Operations

#### Vector Addition

$$
\mathbf{u} + \mathbf{v} = \begin{bmatrix}u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n\end{bmatrix}
$$

Vector addition corresponds to component-wise accumulation.

---

#### Scalar Multiplication

$$
c\mathbf{v} = \begin{bmatrix}cv_1 \\ cv_2 \\ \vdots \\ cv_n\end{bmatrix}
$$

Scalar multiplication represents uniform scaling of a vector.

---

### Matrix Multiplication

Let

$$
\mathbf{A} \in \mathbb{R}^{m \times p}, \quad \mathbf{B} \in \mathbb{R}^{p \times n}
$$

The product is defined symbolically as

$$
\mathbf{C} = \mathbf{A}\mathbf{B}
$$

with elements

$$
c_{ij} = \sum_{k=1}^{p} a_{ik} b_{kj}
$$

This notation reflects composition of linear transformations.

---

### Worked Example

#### Step 1: Define the matrices

$$
\mathbf{A} = \begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}
\qquad
\mathbf{B} = \begin{bmatrix}9 & 8 & 7 \\ 6 & 5 & 4 \\ 3 & 2 & 1\end{bmatrix}
$$

#### Step 2: Define the product

$$
\mathbf{C} = \mathbf{A}\mathbf{B}
$$

#### Step 3: Compute elements

Element-wise arithmetic follows directly from the definition of matrix multiplication.

#### Step 4: Final result

$$
\mathbf{C} = \begin{bmatrix}30 & 24 & 18 \\ 84 & 69 & 54 \\ 138 & 114 & 90\end{bmatrix}
$$

---

### 2. Systems of Linear Equations

A system of linear equations is written compactly as

$$
\mathbf{A}\mathbf{x} = \mathbf{b}
$$

where

$$
\mathbf{A} = \begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}, \quad
\mathbf{x} = \begin{bmatrix}x_1 \\ x_2\end{bmatrix}, \quad
\mathbf{b} = \begin{bmatrix}5 \\ 11\end{bmatrix}
$$

Separating definitions from equations avoids GitHub rendering failures while preserving algebraic meaning.

Row reduction yields

$$
\mathbf{A}_{\text{ref}} = \begin{bmatrix}1 & 2 \\ 0 & -2\end{bmatrix}, \quad
\mathbf{b}_{\text{ref}} = \begin{bmatrix}5 \\ -4\end{bmatrix}
$$

from which

$$
x_2 = 2, \quad x_1 = 1
$$

---

## Part 2: Advanced Linear Algebra for Control Theory

### State-Space Representation

$$
\dot{\mathbf{x}} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u}
$$

$$
\mathbf{y} = \mathbf{C}\mathbf{x} + \mathbf{D}\mathbf{u}
$$

Matrix–vector products are expressed symbolically rather than expanded.

---

### Controllability

The controllability matrix is defined as

$$
\mathbf{C}_c = \begin{bmatrix}\mathbf{B} & \mathbf{A}\mathbf{B} & \cdots & \mathbf{A}^{n-1}\mathbf{B}\end{bmatrix}
$$

Full rank implies reachability of all state directions.

---

### Observability

The observability matrix is

$$
\mathbf{O} = \begin{bmatrix}\mathbf{C} \\ \mathbf{C}\mathbf{A} \\ \vdots \\ \mathbf{C}\mathbf{A}^{n-1}\end{bmatrix}
$$

---

## Part 3: Linear Algebra in AI and Machine Learning

All remaining formulations use symbolic matrix expressions only, which are stable under GitHub rendering.

---

### Final Note to Future-Self

If this document feels dense on re-reading, that is expected.

It is intended as a **map**, not a lesson — a place to reconnect equations to meaning before returning to detailed derivations elsewhere.

