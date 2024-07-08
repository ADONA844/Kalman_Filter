import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 0.1  # Sampling time
total_time = 5
t = np.arange(0, total_time + T, T)  # Time vector
mea = 1  # Number of measurements
I = np.eye(mea)  # Identity Matrix
mean = 0  # Measurement noise mean
meas_noise_sd = 5  # Measurement noise standard deviation
init_pos = np.array([110])  # Initial Position x
F = I  # Transition matrix
H = I  # Output Coefficient Matrix
R = (meas_noise_sd ** 2) * I  # Measurement covariance matrix

# Initialize arrays for storing results
Est_X = np.zeros((50, len(t)))
Znoisyx = np.zeros((50, len(t)))

for m in range(50):  # For 50 runs
    Ztrue = init_pos * np.ones(len(t))
    meas_noise = mean + meas_noise_sd * np.random.randn(len(t))  # Generating measurement noise (Gaussian noise)
    Znoisy = Ztrue + meas_noise  # The noisy measurement vector

    X0 = np.array([[Znoisy[0]]])  # Initial State

    P0 = R  # Initial State Covariance

    for cycle in range(len(t)):
        if cycle == 0:
            X_est = X0  # Initial State
            P_est = P0  # Initial State covariance
        else:
            X_pri = F @ X_est  # Predicted state vector
            P_pri = F @ P_est @ F.T  # Predicted state covariance matrix
            S = H @ P_pri @ H.T + R  # Innovation covariance
            K = P_pri @ H.T @ np.linalg.inv(S)  # Optimal Kalman gain

            X_est = X_pri + K @ (Znoisy[cycle] - H @ X_pri)  # State estimate
            P_est = (I - K @ H) @ P_pri

        Est_X[m, cycle] = X_est
    Znoisyx[m, :] = Znoisy

M_estX = np.mean(Est_X, axis=0)  # Finding means of all runs
M_Znoisyx = np.mean(Znoisyx, axis=0)
RMSE = np.sqrt(np.mean((np.tile(Ztrue, (50, 1)) - Est_X) ** 2, axis=0))  # Root Mean Square error

# Plotting
plt.figure()
plt.plot(t, Ztrue, 'r', linewidth=1.5, label='True Position')
plt.plot(t, M_Znoisyx, 'g', linewidth=1.5, label='Measured Position')
plt.plot(t, M_estX, 'b', linewidth=1.5, label='Estimated Position')
plt.legend()
plt.xlabel('Time in seconds')
plt.ylabel('Position in meters')
plt.title('Target Position')
plt.grid(True)

plt.figure()
plt.plot(t, RMSE, linewidth=1.5)
plt.xlabel('Time in seconds')
plt.ylabel('RMSE in meters')
plt.title('Target Root Mean Square Error')
plt.grid(True)

plt.show()




# NumPy: NumPy is a fundamental package for scientific computing in Python. You should know how to work with NumPy arrays, perform mathematical operations, generate random numbers, and use functions like np.mean(), np.eye(), np.random.randn(), np.linalg.inv(), etc.
#
# Matplotlib: Matplotlib is a plotting library in Python. You should understand how to create different types of plots (line plots, scatter plots, etc.), customize plots (labels, titles, legends, colors), and display multiple plots.
