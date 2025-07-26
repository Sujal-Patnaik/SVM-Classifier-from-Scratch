# 🧠 SVM from Scratch in NumPy

This project implements a Support Vector Machine (SVM) classifier from scratch using only NumPy. It supports **linear**, **polynomial**, and **RBF kernels**, and can be used for **binary** and **multi-class classification** (via one-vs-rest strategy).

---

## 📚 Theoretical Overview

Support Vector Machines aim to find a hyperplane that **maximally separates** classes by maximizing the margin between the support vectors (critical points from each class). It solves the following optimization problem:

\[
\min_{\alpha} \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j) - \sum_i \alpha_i
\]

Subject to:

\[
\sum_i \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C
\]

Here, `K(x_i, x_j)` is the kernel function, allowing non-linear decision boundaries.

The dual problem is solved using the **Sequential Minimal Optimization (SMO)** algorithm.

---

## 🛠️ Features

- Kernel support:
  - Linear
  - Polynomial
  - RBF (Gaussian)
- SMO-based optimization
- Multi-class support (via one-vs-rest)
- Support vector visualization
- From-scratch, no ML libraries used

---

## 📦 Installation

No installation required. Just clone and run:

```bash
git clone https://github.com/yourusername/svm-from-scratch.git
cd svm-from-scratch
python svm_example.py

