# Kalman Filtering

- [ ] Kalman Filtering
    - [ ] Linear Kalman Filter
        - [ ] System model and assumptions
        - [ ] Prediction equations
        - [ ] Update equations
        - [ ] Uncertainty propagation
    - [ ] Extended Kalman Filter (EKF)
        - [ ] Linearization process
        - [ ] Jacobian matrices
        - [ ] Algorithm implementation
    - [ ] Unscented Kalman Filter (UKF)
        - [ ] Sigma points
        - [ ] Unscented transform
        - [ ] Algorithm implementation

## Linear Kalman Filter

### System model and assumptions

At its core, the linear Kalman filter describes a system using two fundamental equations:

1. The State Equation (also called the Process Model):
$$x(k) = F(k)x(k-1) + B(k)u(k) + w(k)$$

This equation tells us how the system evolves over time. Here, $x(k)$ is the current state, $F(k)$ is the state transition matrix that describes how the state naturally evolves, $B(k)u(k)$ represents known control inputs, and $w(k)$ is the process noise.

2. The Measurement Equation (also called the Observation Model):
$$z(k) = H(k)x(k) + v(k)$$

This equation describes how we observe the system. $z(k)$ represents our measurements, $H(k)$ is the measurement matrix that maps the state to measurements, and $v(k)$ is the measurement noise.

Now, let's examine the crucial assumptions that make the linear Kalman filter work:

First, we assume linearity in both equations. This means that both the state transitions and measurements must be linear functions of the state. In real-world terms, if you double the state, the output should double too. This is why it's called a "linear" Kalman filter.

Second, we make important assumptions about the noise terms $w(k)$ and $v(k)$. We assume they are:

- Zero-mean Gaussian white noise
- Uncorrelated with each other
- Uncorrelated in time (white noise)
- Known covariances (Q for process noise and R for measurement noise)

Think of these noise assumptions like rolling fair dice repeatedly - each roll is independent, and while you might not know the exact outcome, you know the probability distribution.

Third, we assume we have some initial knowledge about the state, typically expressed as an initial state estimate $x̂(0)$ and its uncertainty $P(0)$. This doesn't need to be perfectly accurate, but better initial estimates generally lead to faster convergence.

Fourth, we assume that the matrices F(k), H(k), B(k), Q(k), and R(k) are known at each time step. They can vary with time, but we need to know their values.

The beauty of these assumptions is that they allow us to derive optimal state estimates in a mathematically rigorous way. When these assumptions hold true, the Kalman filter gives us the best possible estimate of the state (in terms of minimum mean square error).

### Prediction equations

Let's go through the prediction equations of the Kalman filter. We'll break this down step by step, with a focus on what each equation means practically.

The Prediction Phase (Time Update):
Let's start with predicting where our system will be in the next time step. We have two key equations here:

1. State Prediction:
$$x̂⁻(k) = F(k)x̂(k-1) + B(k)u(k)$$

Where:

- $x̂⁻(k)$ is our predicted state (the superscript minus means "before measurement update")
- $x̂(k-1)$ is our previous state estimate
- $F(k)$ is the state transition matrix
- $B(k)$ is the control input matrix
- $u(k)$ is the control input vector

Think of this like predicting where a moving object will be based on its current position, velocity, and any forces we're applying to it.

2. Error Covariance Prediction:
$$P⁻(k) = F(k)P(k-1)F(k)ᵀ + Q(k)$$

Where:

- $P⁻(k)$ is our predicted error covariance
- $P(k-1)$ is our previous error covariance
- $F(k)ᵀ$ is the transpose of F(k)
- $Q(k)$ is the process noise covariance

This equation tells us how uncertain we are about our prediction. It's crucial for engineering applications because it helps us understand the reliability of our estimates.

Practical Engineering Considerations:

1. State Transition Matrix (F):

For a simple position-velocity system in one dimension, F might look like:

```
F = [1  Δt]
    [0   1]
```

Where Δt is your sampling time. This models constant velocity motion.

2. Process Noise (Q):

For the same system, a common model is:

```
Q = [(Δt⁴/4)σ²  (Δt³/2)σ²]
    [(Δt³/2)σ²     Δt²σ²  ]
```

Where σ² is the variance of your acceleration noise.

3. Control Input:

If you're applying known forces or controls, B(k)u(k) accounts for these. For example, in a rocket system, this might represent thrust.

Engineering Tips:

1. Always check units! Make sure your matrices are dimensionally consistent.
2. When implementing, start with small time steps (Δt) to reduce linearization errors.
3. Q should be tuned based on your system's characteristics - if you're unsure, start with a diagonal matrix with conservative (larger) values.

Common Pitfalls to Avoid:

1. Don't forget to propagate uncertainties - P is just as important as x̂
2. Watch out for numerical issues - consider using square root forms for P if precision is critical
3. Make sure F and Q are properly synchronized with your sampling time

### Update equations

The Measurement Update (Correction Phase) consists of three key equations that work together to incorporate new measurement information into our state estimate:

1. The Kalman Gain Equation:

$$K(k) = P⁻(k)H(k)ᵀ[H(k)P⁻(k)H(k)ᵀ + R(k)]⁻¹$$

This equation determines how much we trust our new measurement versus our prediction. Think of it as a weighting factor that balances between our model and our sensors. Let's break down what each term means:

- $P⁻(k)$ is our predicted error covariance (from the prediction step)
- $H(k)$ is our measurement matrix
- $R(k)$ is our measurement noise covariance
- The $⁻¹$ indicates matrix inversion

2. The State Update Equation:

$$x̂(k) = x̂⁻(k) + K(k)[z(k) - H(k)x̂⁻(k)]$$

This equation updates our state estimate based on the measurement. The term [z(k) - H(k)x̂⁻(k)] is called the innovation or measurement residual - it represents the difference between what we measured and what we expected to measure. We then weight this difference by the Kalman gain to determine how much to correct our prediction.

3. The Error Covariance Update:

$$P(k) = [I - K(k)H(k)]P⁻(k)$$

This equation updates our uncertainty estimate. The term $[I - K(k)H(k)]$ is sometimes called the stability factor, as it helps ensure numerical stability in our computations.

From an engineering perspective, here are some crucial insights about these equations:

The Kalman gain K(k) will automatically adapt based on the relative uncertainties. If R(k) is large (noisy measurements), K(k) will be smaller, meaning we trust our predictions more. If $P⁻(k)$ is large (uncertain predictions), K(k) will be larger, meaning we trust the measurements more.

For implementation, you might encounter alternate forms of the covariance update equation:

- Joseph form: 

$$P(k) = [I - K(k)H(k)]P⁻(k)[I - K(k)H(k)]ᵀ + K(k)R(k)K(k)ᵀ$$

This form is more computationally expensive but numerically more stable. It's particularly useful when dealing with ill-conditioned matrices.

A practical example: Consider a simple position tracking system where we measure position directly. Our measurement matrix might be:

```
H = [1 0]
```

This indicates we're measuring position but not velocity directly. Our measurement noise covariance R might be a single value representing our sensor's variance:

```
R = [σ²_sensor]
```

### Uncertainty propagation

At its core, uncertainty propagation in Kalman filtering deals with how our uncertainty about the state evolves over time and through measurements. This uncertainty is represented by the covariance matrix P, which is a symmetric matrix where the diagonal elements represent variances of individual state components, and off-diagonal elements represent their correlations.

Let's start with linear uncertainty propagation through the prediction step:

$$P⁻(k) = F(k)P(k-1)F(k)ᵀ + Q(k)$$

This equation comes from the linear transformation of random variables. When we have a random variable x and transform it linearly by A, the covariance transforms as APAᵀ. In our case, F is our transformation matrix, and we add Q to account for additional uncertainty from process noise.

To understand this deeper, let's consider a simple 2D state space of position and velocity:

```
P = [σ²_pos       σ_pos_vel  ]
    [σ_pos_vel    σ²_vel     ]
```

When we propagate this through time with:

```
F = [1  Δt]
    [0   1]
```

The uncertainty grows in a specific pattern. The position uncertainty increases due to both:

1. The existing position uncertainty
2. The velocity uncertainty (scaled by Δt)
3. The correlation between position and velocity
4. The process noise Q

This creates a characteristic "banana-shaped" uncertainty region in position-velocity space, because position and velocity become correlated through the prediction step even if they weren't initially.

Moving to the measurement update, uncertainty reduction happens through:

$$P(k) = [I - K(k)H(k)]P⁻(k)$$

This equation shows how incorporating a measurement reduces our uncertainty. The amount of reduction depends on:

1. How uncertain we were before (P⁻)
2. How accurate our measurement is (R)
3. What we're measuring (H)

The Kalman gain K plays a crucial role here. It's derived to minimize the trace of P(k), which means it minimizes the sum of variances of our state estimates. This is why the Kalman filter is called an optimal estimator - it provides the minimum variance estimate given our assumptions.

A practical example helps illustrate this: Imagine tracking a moving object where:

- Position measurements have uncertainty σ_meas = 1m
- Initial velocity uncertainty σ_vel = 0.1 m/s
- Sampling time Δt = 0.1s

Our prediction step would propagate uncertainty like this:

```python
# Initial uncertainty
P = [[1.0, 0.0],    # Position variance = 1m²
     [0.0, 0.01]]   # Velocity variance = 0.01(m/s)²

# State transition
F = [[1.0, 0.1],    # Δt = 0.1s
     [0.0, 1.0]]

# Process noise (simplified)
Q = [[0.001, 0.0],  # Small position process noise
     [0.0, 0.001]]  # Small velocity process noise

# Predicted uncertainty
P_pred = F @ P @ F.T + Q
```

After prediction, you'll notice:

1. Position uncertainty has increased
2. A correlation has developed between position and velocity
3. Velocity uncertainty has grown slightly due to Q

This understanding of uncertainty propagation is crucial for:

1. Filter tuning - setting appropriate Q and R values
2. Sensor fusion - knowing how to weight different measurements
3. System design - understanding how measurement frequency affects estimation quality

## Extended Kalman Filter (EKF)

### Linearization process

Through the Extended Kalman Filter (EKF) linearization process, we adapt the linear Kalman filter concepts to handle nonlinear systems.

In the EKF, we deal with nonlinear system equations:

State Evolution (Process Model):
$$x(k) = f(x(k-1), u(k)) + w(k)$$

Measurement Model:
$$z(k) = h(x(k)) + v(k)$$

Where $f()$ and $h()$ are nonlinear functions. The key insight of the EKF is that we can approximate these nonlinear functions using Taylor series expansion around our current estimate. Let's break this down step by step:

#### Linearization Process {-}
We need to compute Jacobian matrices - these are matrices of partial derivatives that give us the best linear approximation of our nonlinear functions at a specific point. We compute:

$$
F(k) = ∂f/∂x evaluated at x̂(k-1)
H(k) = ∂h/∂x evaluated at x̂⁻(k)
$$

Let's work through a practical example. Consider a robot moving in 2D with state vector:
$$x = [x position, y position, heading angle θ]$$

The nonlinear process model might be:

```python
def f(x, u):
    # x: state [x, y, θ]
    # u: control [velocity, angular_velocity]
    dt = 0.1  # time step
    
    x_new = x[0] + u[0]*cos(x[2])*dt
    y_new = x[1] + u[0]*sin(x[2])*dt
    θ_new = x[2] + u[1]*dt
    
    return np.array([x_new, y_new, θ_new])
```

To linearize this, we compute the Jacobian F:

```python
def compute_F(x, u):
    dt = 0.1
    v = u[0]  # velocity
    
    F = np.array([
        [1, 0, -v*sin(x[2])*dt],
        [0, 1,  v*cos(x[2])*dt],
        [0, 0,  1]
    ])
    return F
```

#### Using the Linearized Models {-}

Once we have our Jacobians, the EKF prediction equations become:

State Prediction:

$$x̂⁻(k) = f(x̂(k-1), u(k))   // Use nonlinear function$$

Covariance Prediction:

$$P⁻(k) = F(k)P(k-1)F(k)ᵀ + Q(k)   // Use linearized F$$

The measurement update is similar:

Innovation:

$$y(k) = z(k) - h(x̂⁻(k))   // Use nonlinear function$$

Kalman Gain:

$$K(k) = P⁻(k)H(k)ᵀ[H(k)P⁻(k)H(k)ᵀ + R(k)]⁻¹   // Use linearized H$$

State Update:

$$x̂(k) = x̂⁻(k) + K(k)y(k)$$

Covariance Update:

$$P(k) = [I - K(k)H(k)]P⁻(k)$$

#### Important Considerations {-}

- The linearization is only valid near the operating point. If your estimate strays too far from the true state, the approximation breaks down.
- You need to recompute the Jacobians at each time step since they depend on the current state estimate.
- Numerical computation of Jacobians can be useful for complex systems:

```python
def numerical_jacobian(f, x, dx=1e-6):
    n = len(x)
    J = np.zeros((n, n))
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += dx
        x_minus = x.copy()
        x_minus[i] -= dx
        J[:, i] = (f(x_plus) - f(x_minus)) / (2*dx)
    return J
```

### Jacobian matrices

The Jacobian matrices in EKF serve as our linear approximation of nonlinear functions. We need two key Jacobians: F(k) for the process model and H(k) for the measurement model.

#### The Process Model Jacobian (F) {-}

F(k) represents how small changes in our state affect the predicted next state. Mathematically, it's the matrix of partial derivatives of our process function f with respect to each state variable:

$$
F(k) = ∂f/∂x = [∂f₁/∂x₁  ∂f₁/∂x₂  ∂f₁/∂x₃ ...]
                [∂f₂/∂x₁  ∂f₂/∂x₂  ∂f₂/∂x₃ ...]
                [∂f₃/∂x₁  ∂f₃/∂x₂  ∂f₃/∂x₃ ...]
                [   ...      ...      ...    ...]
$$

Let's work through a concrete example. Consider a robot moving in 2D with state vector $x = [x, y, θ]$, where:

- $x, y$ are position coordinates
- $θ$ is the heading angle
- The robot moves with velocity v and angular velocity $ω$

The nonlinear process model would be:
$$
x(k+1) = x(k) + v*cos(θ)*dt
y(k+1) = y(k) + v*sin(θ)*dt
θ(k+1) = θ(k) + ω*dt
$$

The Jacobian F would then be:

```python
def compute_process_jacobian(x, v, dt):
    """
    Compute the process model Jacobian for a 2D robot
    x: State vector [x, y, θ]
    v: Linear velocity
    dt: Time step
    """
    F = np.array([
        [1, 0, -v*sin(θ)*dt],  # ∂x'/∂x, ∂x'/∂y, ∂x'/∂θ
        [0, 1,  v*cos(θ)*dt],  # ∂y'/∂x, ∂y'/∂y, ∂y'/∂θ
        [0, 0,  1]             # ∂θ'/∂x, ∂θ'/∂y, ∂θ'/∂θ
    ])
    return F
```

#### The Measurement Model Jacobian (H) {-}
H(k) represents how small changes in our state affect what we expect to measure. It's the matrix of partial derivatives of our measurement function h with respect to each state variable:

$$
H(k) = ∂h/∂x = [∂h₁/∂x₁  ∂h₁/∂x₂  ∂h₁/∂x₃ ...]
                [∂h₂/∂x₁  ∂h₂/∂x₂  ∂h₂/∂x₃ ...]
                [   ...      ...      ...    ...]
$$


For example, if we're measuring range and bearing to a landmark at position $[xₗ, yₗ]$:
$$r = √((x - xₗ)² + (y - yₗ)²)$$
$$β = atan2(yₗ - y, xₗ - x) - θ$$

The measurement Jacobian would be:

```python
def compute_measurement_jacobian(x, landmark_pos):
    """
    Compute measurement Jacobian for range-bearing measurements
    x: State vector [x, y, θ]
    landmark_pos: Position of landmark [xₗ, yₗ]
    """
    dx = landmark_pos[0] - x[0]
    dy = landmark_pos[1] - x[1]
    r = np.sqrt(dx**2 + dy**2)
    
    H = np.array([
        [-dx/r,      -dy/r,       0    ],  # ∂r/∂x,  ∂r/∂y,  ∂r/∂θ
        [dy/r**2,    -dx/r**2,   -1    ]   # ∂β/∂x,  ∂β/∂y,  ∂β/∂θ
    ])
    return H
```

Important Considerations for Jacobian Computation:

1. Singularity Points: Some configurations might lead to undefined Jacobians. For example, in our range-bearing case, when the robot is exactly at the landmark position (r = 0), the bearing Jacobian becomes undefined.

2. Numerical Stability: When implementing Jacobians, watch out for numerical issues. For very small values, you might need to add small constants to denominators:
```python
def safe_division(num, den, eps=1e-10):
    return num / (den + eps)
```

3. Verification: You can verify your analytical Jacobians using numerical differentiation:

```python
def verify_jacobian(f, x, dx=1e-6):
    """
    Verify analytical Jacobian against numerical approximation
    """
    analytical = compute_jacobian(x)
    numerical = numerical_jacobian(f, x, dx)
    difference = np.abs(analytical - numerical)
    print("Maximum difference:", np.max(difference))
```

### Algorithm implementation

Let's create a comprehensive implementation of both Kalman Filter and Extended Kalman Filter with detailed documentation and test scenarios. The key components of this implementation are:

#### Base Classes: {-}

- `KalmanFilterState`: A dataclass to store filter states
- `KalmanFilter`: Implementation of the linear Kalman filter
- `ExtendedKalmanFilter`: Implementation of the EKF

#### Test Scenarios {-}

- Linear case: Tracking an object moving with constant velocity
- Nonlinear case: Tracking a robot with nonlinear dynamics using range-bearing measurements

The code includes detailed comments explaining each component and step.

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import matplotlib.pyplot as plt

@dataclass
class KalmanFilterState:
    """
    Dataclass to store the state of a Kalman Filter
    
    Attributes:
        x (np.ndarray): State estimate vector
        P (np.ndarray): State covariance matrix
        dim_x (int): Dimension of state vector
        dim_z (int): Dimension of measurement vector
    """
    x: np.ndarray  # State estimate
    P: np.ndarray  # Covariance matrix
    dim_x: int     # State dimension
    dim_z: int     # Measurement dimension

class KalmanFilter:
    """
    Implementation of a linear Kalman Filter.
    
    This implementation follows the standard Kalman Filter equations for
    linear systems with Gaussian noise.
    """
    
    def __init__(self, dim_x: int, dim_z: int):
        """
        Initialize Kalman Filter
        
        Args:
            dim_x: Dimension of state vector
            dim_z: Dimension of measurement vector
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # Initialize matrices
        self.x = np.zeros(dim_x)         # State estimate
        self.P = np.eye(dim_x)           # State covariance
        self.F = np.eye(dim_x)           # State transition matrix
        self.H = np.zeros((dim_z, dim_x))# Measurement matrix
        self.R = np.eye(dim_z)           # Measurement noise covariance
        self.Q = np.eye(dim_x)           # Process noise covariance
        
    def predict(self, u: Optional[np.ndarray] = None, B: Optional[np.ndarray] = None) -> None:
        """
        Predict step of the Kalman Filter
        
        Args:
            u: Optional control input
            B: Optional control matrix
        """
        # State prediction
        if u is not None and B is not None:
            self.x = self.F @ self.x + B @ u
        else:
            self.x = self.F @ self.x
            
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z: np.ndarray) -> None:
        """
        Update step of the Kalman Filter
        
        Args:
            z: Measurement vector
        """
        # Innovation and innovation covariance
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State and covariance update
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P
        
    def get_state(self) -> KalmanFilterState:
        """Return current filter state"""
        return KalmanFilterState(
            x=self.x.copy(),
            P=self.P.copy(),
            dim_x=self.dim_x,
            dim_z=self.dim_z
        )

class ExtendedKalmanFilter:
    """
    Implementation of an Extended Kalman Filter.
    
    This implementation handles nonlinear systems by linearizing
    around the current state estimate.
    """
    
    def __init__(self, dim_x: int, dim_z: int,
                 f: Callable, h: Callable,
                 compute_F: Callable, compute_H: Callable):
        """
        Initialize Extended Kalman Filter
        
        Args:
            dim_x: Dimension of state vector
            dim_z: Dimension of measurement vector
            f: Nonlinear state transition function
            h: Nonlinear measurement function
            compute_F: Function to compute state transition Jacobian
            compute_H: Function to compute measurement Jacobian
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # Nonlinear functions
        self.f = f          # State transition function
        self.h = h          # Measurement function
        self.compute_F = compute_F  # State transition Jacobian
        self.compute_H = compute_H  # Measurement Jacobian
        
        # Initialize matrices
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.R = np.eye(dim_z)
        self.Q = np.eye(dim_x)
        
    def predict(self, u: Optional[np.ndarray] = None) -> None:
        """
        Predict step of the EKF
        
        Args:
            u: Optional control input
        """
        # Compute state transition Jacobian
        F = self.compute_F(self.x, u)
        
        # Nonlinear state prediction
        if u is not None:
            self.x = self.f(self.x, u)
        else:
            self.x = self.f(self.x, None)
            
        # Covariance prediction using linearized F
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, z: np.ndarray) -> None:
        """
        Update step of the EKF
        
        Args:
            z: Measurement vector
        """
        # Compute measurement Jacobian
        H = self.compute_H(self.x)
        
        # Innovation using nonlinear measurement function
        y = z - self.h(self.x)
        
        # Innovation covariance using linearized H
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State and covariance update
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ H) @ self.P
        
    def get_state(self) -> KalmanFilterState:
        """Return current filter state"""
        return KalmanFilterState(
            x=self.x.copy(),
            P=self.P.copy(),
            dim_x=self.dim_x,
            dim_z=self.dim_z
        )

# Test scenarios

def test_linear_constant_velocity():
    """
    Test scenario: Track an object moving with constant velocity
    using a linear Kalman filter
    """
    # Initialize Kalman Filter
    kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, vx, vy], Measurement: [x, y]
    dt = 0.1
    
    # Set up system matrices
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    # Set noise covariances
    kf.R = np.eye(2) * 0.1  # Measurement noise
    kf.Q = np.eye(4) * 0.01  # Process noise
    
    # Generate true trajectory
    t = np.arange(0, 10, dt)
    true_x = 0.1 * t**2  # Quadratic motion in x
    true_y = 0.1 * t     # Linear motion in y
    
    # Generate noisy measurements
    measurements = np.column_stack((
        true_x + np.random.normal(0, 0.1, len(t)),
        true_y + np.random.normal(0, 0.1, len(t))
    ))
    
    # Run filter
    estimated_states = []
    for z in measurements:
        kf.predict()
        kf.update(z)
        estimated_states.append(kf.get_state().x.copy())
    
    estimated_states = np.array(estimated_states)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.plot(true_x, true_y, 'k-', label='True')
    plt.plot(measurements[:, 0], measurements[:, 1], 'r.', label='Measurements')
    plt.plot(estimated_states[:, 0], estimated_states[:, 1], 'b--', label='Estimated')
    plt.legend()
    plt.title('Position Track')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.subplot(122)
    plt.plot(t, estimated_states[:, 2], 'r-', label='Estimated Vx')
    plt.plot(t, estimated_states[:, 3], 'b-', label='Estimated Vy')
    plt.legend()
    plt.title('Velocity Estimates')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    
    plt.tight_layout()
    plt.show()

def test_nonlinear_robot():
    """
    Test scenario: Track a robot moving with nonlinear dynamics
    using an Extended Kalman Filter
    """
    def f(x, u):
        """Nonlinear state transition function"""
        dt = 0.1
        if u is None:
            v = 1.0  # Constant velocity if no control
            omega = 0.1  # Constant angular velocity
        else:
            v = u[0]
            omega = u[1]
            
        theta = x[2]
        return np.array([
            x[0] + v * np.cos(theta) * dt,
            x[1] + v * np.sin(theta) * dt,
            x[2] + omega * dt
        ])
    
    def h(x):
        """Nonlinear measurement function (range and bearing to landmark)"""
        landmark = np.array([5.0, 5.0])  # Fixed landmark position
        dx = landmark[0] - x[0]
        dy = landmark[1] - x[1]
        
        r = np.sqrt(dx**2 + dy**2)
        bearing = np.arctan2(dy, dx) - x[2]
        return np.array([r, bearing])
    
    def compute_F(x, u):
        """Compute state transition Jacobian"""
        dt = 0.1
        if u is None:
            v = 1.0
        else:
            v = u[0]
            
        theta = x[2]
        return np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1,  v * np.cos(theta) * dt],
            [0, 0, 1]
        ])
    
    def compute_H(x):
        """Compute measurement Jacobian"""
        landmark = np.array([5.0, 5.0])
        dx = landmark[0] - x[0]
        dy = landmark[1] - x[1]
        r = np.sqrt(dx**2 + dy**2)
        
        return np.array([
            [-dx/r, -dy/r, 0],
            [dy/r**2, -dx/r**2, -1]
        ])
    
    # Initialize EKF
    ekf = ExtendedKalmanFilter(
        dim_x=3,  # State: [x, y, theta]
        dim_z=2,  # Measurement: [range, bearing]
        f=f,
        h=h,
        compute_F=compute_F,
        compute_H=compute_H
    )
    
    # Set noise covariances
    ekf.R = np.diag([0.1, 0.1])  # Measurement noise
    ekf.Q = np.diag([0.01, 0.01, 0.01])  # Process noise
    
    # Generate true trajectory
    t = np.arange(0, 10, 0.1)
    true_states = []
    x = np.array([0.0, 0.0, 0.0])  # Initial state
    
    for _ in t:
        true_states.append(x.copy())
        x = f(x, None)  # Use constant velocity and angular velocity
    
    true_states = np.array(true_states)
    
    # Generate noisy measurements
    measurements = []
    for state in true_states:
        z = h(state)
        z += np.random.normal(0, np.sqrt(np.diag(ekf.R)))
        measurements.append(z)
    
    measurements = np.array(measurements)
    
    # Run filter
    estimated_states = []
    for z in measurements:
        ekf.predict()
        ekf.update(z)
        estimated_states.append(ekf.get_state().x.copy())
    
    estimated_states = np.array(estimated_states)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.plot(true_states[:, 0], true_states[:, 1], 'k-', label='True')
    plt.plot(estimated_states[:, 0], estimated_states[:, 1], 'r--', label='Estimated')
    plt.plot(5.0, 5.0, 'g*', markersize=10, label='Landmark')
    plt.legend()
    plt.title('Robot Position Track')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.subplot(122)
    plt.plot(t, np.unwrap(true_states[:, 2]), 'k-', label='True Heading')
    plt.plot(t, np.unwrap(estimated_states[:, 2]), 'r--', label='Estimated Heading')
    plt.legend()
    plt.title('Robot Heading')
    plt.xlabel('Time')
    plt.ylabel('Heading (rad)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Running Linear Kalman Filter test...")
    test_linear_constant_velocity()
    
    print("\nRunning Extended Kalman Filter test...")
    test_nonlinear_robot()
```
## Unscented Kalman Filter (UKF)

### Sigma points

### Unscented transform

### Algorithm implementation