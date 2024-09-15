import numpy as np
import matplotlib.pyplot as plt


def generation(length):
    sequence = np.zeros(length)
    sequence[0] = 1
    for k in range(1, length):
        rv = np.random.random()
        sequence[k] = 0.2 * sequence[k - 1] + np.sin(k) + rv
    return sequence


length = 100
Zi = generation(length)

# Modified matrices
A = np.array([[0.2]])  # State Transition Matrix
B = np.array([[1]])  # Control Input Matrix
Q = np.array([[1]])  # Process Noise Covariance Matrix
R = np.array([[1]])  # Measurement Noise Covariance Matrix
H = np.array([[1]])  # Measurement Matrix
P = np.array([[1]])  # Error Covariance Matrix

xi = np.array([[0]])  # Initial state estimate
estimated_states = np.zeros(length)

for k in range(length):
    u = np.array([[np.sin(k)]])

    # Prediction step
    x = A * xi + B * u
    P = A * P * A.T + Q

    # Measurement Update
    K = P * H.T @ np.linalg.inv(H * P * H.T + R)
    xi = x + K * (Zi[k] - H * x)
    P = (np.eye(1) - K * H) * P
    estimated_states[k] = xi.flatten()

true_state = Zi
estimated_state = estimated_states

plt.figure(figsize=(12, 6))
plt.plot(true_state, label='True State')
plt.plot(np.arange(length), estimated_state, label='Estimated State', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.title('True State vs. Estimated State')
plt.legend()
plt.grid()
plt.show()
