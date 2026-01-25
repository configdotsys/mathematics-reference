# Cramer's Rule

Cramer's Rule is a direct method for solving a system of **n linear equations** in **n unknowns** using **determinants**.

---

## General Form

Given a system of equations in matrix form:

```math
A \cdot x = b
```

Where:
- $A$ is an $n \times n$ matrix of coefficients,
- $x$ is the column vector of unknowns $[x_1, x_2, ..., x_n]^T$,
- $b$ is the column vector of constants $[b_1, b_2, ..., b_n]^T$.

Then each variable $x_k$ is given by:

```math
x_k = \frac{\det(A_k)}{\det(A)}
```

Where:
- $A_k$ is the matrix $A$ with the **k-th column replaced** by the vector $b$,
- $\det(A)$ must be **nonzero** for a unique solution to exist.

---

## 2×2 System Example

Given the system:

```math
\begin{cases}
a_{11} \cdot x + a_{12} \cdot y = b_1 \\
a_{21} \cdot x + a_{22} \cdot y = b_2
\end{cases}
```

Matrix form:

```math
\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}
```

### Step 1: Compute det(A)

```math
\det(A) = a_{11} \cdot a_{22} - a_{12} \cdot a_{21}
```

### Step 2: Solve for x

Replace column 1 of A with b:

```math
\begin{bmatrix} b_1 & a_{12} \\ b_2 & a_{22} \end{bmatrix}
```

Then:

```math
x = \frac{b_1 \cdot a_{22} - a_{12} \cdot b_2}{\det(A)}
```

### Step 3: Solve for y

Replace column 2 of A with b:

```math
\begin{bmatrix} a_{11} & b_1 \\ a_{21} & b_2 \end{bmatrix}
```

Then:

```math
y = \frac{a_{11} \cdot b_2 - b_1 \cdot a_{21}}{\det(A)}
```

---

## Cramer's Rule for a 3×3 System

Cramer's Rule provides a method to solve a system of linear equations using determinants. For a 3×3 system:

### 1. General Form

Given:

```math
\begin{cases}
a_1x + b_1y + c_1z = d_1 \\
a_2x + b_2y + c_2z = d_2 \\
a_3x + b_3y + c_3z = d_3
\end{cases}
```

This is written in matrix form as:

```math
A\vec{x} = \vec{d}
```

Where:

```math
A = \begin{bmatrix} a_1 & b_1 & c_1 \\ a_2 & b_2 & c_2 \\ a_3 & b_3 & c_3 \end{bmatrix}
```

```math
\vec{x} = \begin{bmatrix} x \\ y \\ z \end{bmatrix}
```

```math
\vec{d} = \begin{bmatrix} d_1 \\ d_2 \\ d_3 \end{bmatrix}
```

---

### 2. Cramer's Rule

If $\det(A) \neq 0$, then:

```math
x = \frac{\det(A_x)}{\det(A)}, \quad
y = \frac{\det(A_y)}{\det(A)}, \quad
z = \frac{\det(A_z)}{\det(A)}
```

Where:
- $A_x$ is formed by replacing the first column of $A$ with $\vec{d}$
- $A_y$ is formed by replacing the second column of $A$ with $\vec{d}$
- $A_z$ is formed by replacing the third column of $A$ with $\vec{d}$

---

### 3. Determinant of a 3×3 Matrix

For any 3×3 matrix:

```math
M = \begin{bmatrix}
m_{11} & m_{12} & m_{13} \\
m_{21} & m_{22} & m_{23} \\
m_{31} & m_{32} & m_{33}
\end{bmatrix}
```

The determinant is computed as:

```math
\det(M) = m_{11}(m_{22}m_{33} - m_{23}m_{32}) - m_{12}(m_{21}m_{33} - m_{23}m_{31}) + m_{13}(m_{21}m_{32} - m_{22}m_{31})
```

---

### 4. Worked Example

Solve the system:

```math
\begin{cases}
2x + 3y + z = 1 \\
4x + y + 5z = 25 \\
3x + 2y + 4z = 18
\end{cases}
```

#### Step 1: Define matrices

```math
A = \begin{bmatrix}
2 & 3 & 1 \\
4 & 1 & 5 \\
3 & 2 & 4
\end{bmatrix}, \quad
\vec{d} = \begin{bmatrix}
1 \\
25 \\
18
\end{bmatrix}
```

#### Step 2: Compute $\det(A)$

```math
\det(A) = 2(1 \cdot 4 - 5 \cdot 2) - 3(4 \cdot 4 - 5 \cdot 3) + 1(4 \cdot 2 - 1 \cdot 3)
```

```math
= 2(4 - 10) - 3(16 - 15) + (8 - 3) = -12 - 3 + 5 = -10
```

#### Step 3: Construct modified matrices

**Replace columns with $\vec{d}$:**

```math
A_x = \begin{bmatrix} 1 & 3 & 1 \\ 25 & 1 & 5 \\ 18 & 2 & 4 \end{bmatrix}
```

```math
A_y = \begin{bmatrix} 2 & 1 & 1 \\ 4 & 25 & 5 \\ 3 & 18 & 4 \end{bmatrix}
```

```math
A_z = \begin{bmatrix} 2 & 3 & 1 \\ 4 & 1 & 25 \\ 3 & 2 & 18 \end{bmatrix}
```

#### Step 4: Compute determinants

**$\det(A_x)$:**

```math
= 1(1 \cdot 4 - 5 \cdot 2) - 3(25 \cdot 4 - 5 \cdot 18) + 1(25 \cdot 2 - 1 \cdot 18)
```

```math
= (4 - 10) - 3(100 - 90) + (50 - 18) = -6 - 30 + 32 = -4
```

**$\det(A_y)$:**

```math
= 2(25 \cdot 4 - 5 \cdot 18) - 1(4 \cdot 4 - 5 \cdot 3) + 1(4 \cdot 18 - 25 \cdot 3)
```

```math
= 2(100 - 90) - (16 - 15) + (72 - 75) = 20 - 1 - 3 = 16
```

**$\det(A_z)$:**

```math
= 2(1 \cdot 18 - 25 \cdot 2) - 3(4 \cdot 18 - 25 \cdot 3) + 1(4 \cdot 2 - 1 \cdot 3)
```

```math
= 2(18 - 50) - 3(72 - 75) + (8 - 3) = -64 + 9 + 5 = -50
```

---

#### Step 5: Solve using Cramer's Rule

```math
x = \frac{-4}{-10} = 0.4, \quad
y = \frac{16}{-10} = -1.6, \quad
z = \frac{-50}{-10} = 5
```

---

### Final Answer

```math
\boxed{
x = 0.4,\quad
y = -1.6,\quad
z = 5
}
```

---

## When to Use

Use Cramer's Rule when:
- You have a **2×2 or 3×3** system
- You need to solve for just **one variable**
- You're working with **symbolic algebra**

Avoid Cramer's Rule when:
- The matrix is large ($n > 3$)
- The determinant is **zero** (no unique solution)
- Numerical stability is a concern (use Gaussian elimination instead)

---

## Summary

To solve for $x_k$:

1. Compute $\det(A)$
2. Replace the k-th column of $A$ with vector $b$ → get $A_k$
3. Compute $\det(A_k)$
4. Use: $x_k = \det(A_k) / \det(A)$
