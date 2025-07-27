Here's a detailed and well-formatted `README.md` file for your SVM implementation from scratch. This version includes a comprehensive explanation, usage instructions, and LaTeX-rendered math via GitHub-compatible Markdown.

---

## 📘 Support Vector Machine (SVM) from Scratch

This repository provides a **from-scratch implementation of Support Vector Machines (SVMs)** using only NumPy, supporting both:

* **Sequential Minimal Optimization (SMO)** algorithm for training with custom kernels
* **Gradient Descent (GD)** training for linear SVMs (with Hinge loss)

### 📂 File Structure

* `SVM` class: Contains all necessary methods for training and prediction using SMO and GD.
* Kernels: Linear, Polynomial, and RBF kernels supported.

---

## 🧠 Theory Behind SVM

### Objective

SVMs aim to find the hyperplane that maximally separates two classes. This is achieved by solving the following optimization problem:

$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1
$$

When the data is not linearly separable, the soft-margin SVM is used:

$$
\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i \quad \text{subject to} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

The **dual form** of this optimization (used in SMO) is:

$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)
$$

subject to:

$$
0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{n} \alpha_i y_i = 0
$$

---

## ⚙️ Class Overview

### `__init__`

Initializes the SVM with user-defined:

* `kernel` – one of `'linear'`, `'polynomial'`, or `'rbf'`
* `C` – regularization parameter
* `tol` – tolerance for stopping criterion
* `max_passes` – max number of passes without alpha updates (for SMO)
* `max_iter` – total iteration limit (for SMO)
* `degree` – for polynomial kernel
* `gamma` – for RBF kernel

---

## 🔧 Kernels

```python
def kernel(self, x, y):
```

Defines:

* **Linear kernel**: $K(x, y) = x^T y$
* **Polynomial kernel**: $K(x, y) = (x^T y + 1)^d$
* **RBF kernel**:

$$
K(x, y) = \exp\left(-\gamma \|x - y\|^2\right)
$$

---

## 🧮 Training with SMO

```python
def fit(self, X, y):
```

This is the core of the SVM training process using the **Sequential Minimal Optimization (SMO)** algorithm. The `fit` method is responsible for solving the dual optimization problem by iteratively updating the Lagrange multipliers `α` and the bias `b`.

### 🔍 What Happens Inside?

1. **Initialization**:
   - All Lagrange multipliers `α_i` are initialized to zero.
   - The bias `b` is initialized to zero.
   - The input labels `y` are transformed from `{0, 1}` to `{-1, 1}` to comply with the SVM formulation.

2. **Main SMO Loop**:
   - The algorithm looks for a pair of `α_i` and `α_j` that violate the Karush-Kuhn-Tucker (KKT) conditions. These are candidates for update.
   - For each training point `i`, the decision function is evaluated:

     $$
     f(x_i) = \sum_{j=1}^{n} \alpha_j y_j K(x_j, x_i) + b
     $$

     The error for point `i` is then computed as:

     $$
     E_i = f(x_i) - y_i
     $$

3. **Selecting `α_j`**:
   - Once `i` is chosen, another index `j ≠ i` is selected heuristically (often at random).
   - Similarly, we calculate the error `E_j`.

4. **Computing Update Parameters**:
   - The kernel function is used to compute:

     $$
     \eta = 2 K(x_i, x_j) - K(x_i, x_i) - K(x_j, x_j)
     $$

   - If `η ≥ 0`, we skip the update (as it would not improve the objective function).
   - Otherwise, the new `α_j` is updated using:

     $$
     \alpha_j^{\text{new}} = \alpha_j^{\text{old}} - \frac{y_j (E_i - E_j)}{\eta}
     $$

5. **Clipping α_j**:
   - The new value of `α_j` must be clipped within `[L, H]`, where:
     - If `y_i ≠ y_j`:

       $$
       L = \max(0, \alpha_j - \alpha_i), \quad H = \min(C, C + \alpha_j - \alpha_i)
       $$

     - If `y_i = y_j`:

       $$
       L = \max(0, \alpha_i + \alpha_j - C), \quad H = \min(C, \alpha_i + \alpha_j)
       $$

6. **Updating α_i**:
   - Once `α_j` is updated, `α_i` is computed as:

     $$
     \alpha_i^{\text{new}} = \alpha_i^{\text{old}} + y_i y_j (\alpha_j^{\text{old}} - \alpha_j^{\text{new}})
     $$

7. **Updating Bias Term `b`**:
   - Two possible updated bias values are computed:

     $$
     b_1 = b - E_i - y_i (α_i^{\text{new}} - α_i^{\text{old}}) K(x_i, x_i) - y_j (α_j^{\text{new}} - α_j^{\text{old}}) K(x_i, x_j)
     $$

     $$
     b_2 = b - E_j - y_i (α_i^{\text{new}} - α_i^{\text{old}}) K(x_i, x_j) - y_j (α_j^{\text{new}} - α_j^{\text{old}}) K(x_j, x_j)
     $$

   - The final `b` is chosen depending on whether `α_i` or `α_j` lies strictly between 0 and `C`. If both lie at the bounds, the average of `b1` and `b2` is taken.

8. **Repeat Until Convergence**:
   - The process continues until a certain number of passes occur without any updates to `α` (defined by `max_passes`), or until the maximum number of iterations is reached (`max_iter`).

---

### 🧠 Why This Works

The SMO algorithm effectively solves the **quadratic programming problem** underlying the SVM by breaking it into 2-dimensional sub-problems, which are easier to solve analytically. This allows us to train non-linear SVMs efficiently with custom kernels — without relying on external optimization libraries.

---

### 🧪 Implementation Tips

- You can monitor convergence by tracking how many α values change in each pass.
- Large values of `C` lead to smaller margins and more aggressive fitting.
- For RBF or polynomial kernels, tuning `gamma` and `degree` respectively is essential for good performance.

---

### 🛑 Limitations

- SMO works best for smaller datasets; it can become slow for very large ones.
- For large-scale linear problems, the `fit_gd` method is more efficient.

---

Let me know if you’d like me to break this into docstrings inside your actual `fit` method, or include visual examples of how alphas update.


---

## 🧠 Decision Function (SMO)

```python
def _decision_function_index(self, i):
```

Computes the decision function for the $i^{th}$ training point:

$$
f(x_i) = \sum_{j=1}^{n} \alpha_j y_j K(x_j, x_i) + b
$$

---

## 📊 Prediction (SMO)

```python
def predict(self, X):
```

Projects test data and returns the class prediction:

$$
\hat{y} = \text{sign} \left( \sum_{i \in SV} \alpha_i y_i K(x_i, x) + b \right)
$$

---

## 🔁 Training with Gradient Descent

```python
def fit_gd(self, X, y, lr=0.001, epochs=1000):
```

* Vectorized training for linear SVM
* Optimizes **hinge loss**:

$$
\mathcal{L}(\mathbf{w}, b) = \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))
$$

Gradient updates:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \left( \mathbf{w} - C \sum_{i: y_i f(x_i) < 1} y_i x_i \right)
$$

$$
b \leftarrow b - \eta \left( - C \sum_{i: y_i f(x_i) < 1} y_i \right)
$$

---

## 🔮 Prediction (Gradient Descent)

```python
def predict_gd(self, X):
```

Returns the sign of the linear model output:

$$
\hat{y} = \text{sign}(X \cdot \mathbf{w} + b)
$$

---

## ✅ Usage Example

```python
X = np.array([[1,2], [2,3], [3,3], [2,1], [3,2]])
y = np.array([1, 1, 1, 0, 0])

model = SVM(kernel='rbf', C=1.0)
model.fit(X, y)

preds = model.predict(X)
print(preds)
```

For linear SVM with gradient descent:

```python
model = SVM(C=0.1)
model.fit_gd(X, y)
preds = model.predict_gd(X)
```

---

## 🧪 Testing

Try running the model on toy datasets like XOR, circles, or linearly separable sets to see the effect of kernels.

---

## 📌 Notes

* **Gradient Descent** method supports only **linear kernel**
* **SMO** can handle **non-linear kernels** like RBF and polynomial
* Both methods convert label `0` to `-1` to comply with SVM formulation

---

## 🧑‍💻 Author

Implemented from scratch by \[Your Name or Team Name].

---

## 📄 License

This project is licensed under the MIT License - feel free to use and modify.

---

Let me know if you want me to add visualizations or dataset demo scripts too.
