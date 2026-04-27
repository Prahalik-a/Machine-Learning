import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# basic ops
def summation(x, w, b):
    return np.dot(x, w) + b

def step(x):
    return 1 if x >= 0 else 0

def bipolar_step(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return max(0, x)

def leaky_relu(x):
    return x if x > 0 else 0.01 * x

def comparator(y, target):
    return y - target


# perceptron training
def train_perceptron(X, y, activation, lr=0.05, epochs=1000):
    w = np.random.uniform(-0.5, 0.5, X.shape[1])
    b = np.random.uniform(-0.5, 0.5)
    errors = []

    for epoch in range(epochs):
        total_error = 0

        for i in range(len(X)):
            net = summation(X[i], w, b)
            out = activation(net)

            err = comparator(out, y[i])
            total_error += err**2

            w = w - lr * err * X[i]
            b = b - lr * err

        mse = total_error / len(X)
        errors.append(mse)

        if mse <= 0.002:
            break

    return w, b, errors, epoch + 1


# logic data
X_logic = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y_and = np.array([0,0,0,1])
y_xor = np.array([0,1,1,0])


# AND using step
w_and, b_and, err_and, ep_and = train_perceptron(X_logic, y_and, step)

plt.plot(err_and)
plt.title("AND Error")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()


# different activations
activations = {
    "Bipolar": bipolar_step,
    "Sigmoid": sigmoid,
    "ReLU": relu
}

results_A3 = {}

for name, func in activations.items():
    w, b, err, ep = train_perceptron(X_logic, y_and, func)
    results_A3[name] = ep


# learning rate test
learning_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
epochs_lr = []

for lr in learning_rates:
    _, _, _, ep = train_perceptron(X_logic, y_and, step, lr=lr)
    epochs_lr.append(ep)

plt.plot(learning_rates, epochs_lr, marker='o')
plt.title("LR vs Epochs")
plt.xlabel("Learning Rate")
plt.ylabel("Epochs")
plt.show()


# XOR (won’t converge)
w_xor, b_xor, err_xor, ep_xor = train_perceptron(X_logic, y_xor, step)


# customer dataset
X_cust = np.array([
    [20,6,2,386],
    [16,3,6,289],
    [27,6,2,393],
    [19,1,2,110],
    [24,4,2,280],
    [22,1,5,167],
    [15,4,2,271],
    [18,4,2,274],
    [21,1,4,148],
    [16,2,4,198]
])

y_cust = np.array([1,1,1,0,1,0,1,1,0,0])

# simple normalization
def normalize(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

X_cust = normalize(X_cust)

w_cust, b_cust, err_cust, ep_cust = train_perceptron(X_cust, y_cust, sigmoid)


# pseudo inverse solution
def pseudo_inverse(X, y):
    X_bias = np.c_[np.ones(len(X)), X]
    w = np.linalg.pinv(X_bias).dot(y)
    return w

w_pinv = pseudo_inverse(X_cust, y_cust)


# basic MLP from scratch
def train_mlp(X, y, lr=0.05, epochs=1000):
    np.random.seed(0)

    W1 = np.random.uniform(-0.05,0.05,(2,2))
    W2 = np.random.uniform(-0.05,0.05,(2,1))

    errors = []

    for epoch in range(epochs):
        total_error = 0

        for i in range(len(X)):
            x = X[i].reshape(1,-1)
            target = y[i]

            # forward pass
            h = sigmoid(np.dot(x, W1))
            o = sigmoid(np.dot(h, W2))

            error = target - o
            total_error += error**2

            # backprop
            d_o = error * o * (1 - o)
            d_h = d_o.dot(W2.T) * h * (1 - h)

            W2 += lr * h.T.dot(d_o)
            W1 += lr * x.T.dot(d_h)

        mse = total_error / len(X)
        errors.append(mse)

        if mse <= 0.002:
            break

    return W1, W2, errors


# MLP for AND
W1, W2, err_mlp = train_mlp(X_logic, y_and)

# MLP for XOR
W1_xor, W2_xor, err_xor_mlp = train_mlp(X_logic, y_xor)


# encode outputs (one-hot)
def encode_output(y):
    return np.array([[1,0] if val==0 else [0,1] for val in y])

y_and_encoded = encode_output(y_and)


# sklearn MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(4,),
    solver='lbfgs',
    activation='logistic',
    max_iter=1000,
    random_state=0
)
mlp.fit(X_logic, y_and)


# project model
mlp_proj = MLPClassifier(
    hidden_layer_sizes=(6,),
    solver='lbfgs',
    max_iter=1000,
    random_state=0
)
mlp_proj.fit(X_cust, y_cust)