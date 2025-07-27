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

* Implements **SMO algorithm**
* Maintains alpha values and bias $b$
* Uses heuristics to choose optimal alpha pairs
* Updates alphas and $b$ using:

$$
\eta = 2 K(x_i, x_j) - K(x_i, x_i) - K(x_j, x_j)
$$

$$
\alpha_j^{\text{new}} = \alpha_j^{\text{old}} - \frac{y_j (E_i - E_j)}{\eta}
$$

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
