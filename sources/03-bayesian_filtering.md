# Bayesian Filtering Framework

- [ ] Bayesian Filtering Framework
    - [ ] Recursive state estimation
    - [ ] Prediction step (motion model)
    - [ ] Update step (measurement model)
    - [ ] Chapman-Kolmogorov equation
    - [ ] Bayes filter algorithm
    - [ ] Linear vs nonlinear systems

## Recursive state estimation

At its heart, recursive state estimation is about continuously updating our belief about a system's state as new measurements come in. Imagine you're trying to track a moving object - at any moment, you want to know its position and velocity, but your sensors are noisy and the object's movement isn't perfectly predictable.

The recursive nature comes from how we update our estimate: instead of processing all past measurements every time we get new data, we maintain a current estimate and update it using only the newest measurement. This makes the process computationally efficient and suitable for real-time applications.

The mathematical framework follows two main steps that repeat over time:

1. Prediction Step (Time Update):
First, we predict how the state will evolve based on our system model:

$$ p(x_{k}|z_{1:k-1})=\int p(x_{k}|x_{k-1})p(x_{k-1}|z_{1:k-1})dx_{k-1}$$

This integral combines our previous estimate $p(x_{k-1}|z_{1:k-1})$ with our motion model $p(x_{k}|x_{k-1})$ to predict the new state.

2. Correction Step (Measurement Update):
When we get a new measurement, we update our prediction using Bayes' rule:

$$ p(x_{k}|z_{1:k})=\eta p(z_{k}|x_{k})p(x_{k}|z_{1:k-1})$$

Here, $p(z_{k}|x_{k})$ is our measurement model, and η is a normalizing constant.

Let's make this concrete with an example. Imagine you're tracking a drone:
- State (x): position and velocity
- Measurements (z): GPS readings
- Motion model: physics equations for how the drone moves
- Measurement model: GPS error characteristics

At each time step:
1. You predict where the drone should be based on physics and your last estimate
2. You get a GPS reading
3. You combine your prediction with the measurement to get an updated estimate
4. Repeat for the next time step

The beauty of this approach is that it naturally handles uncertainty. Both your predictions and measurements come with uncertainty (represented as probability distributions), and the Bayesian framework tells you exactly how to combine these uncertainties to get your best estimate.

A key insight is that this framework maintains a complete probability distribution over possible states, not just a single "best guess." This gives you important information about how certain you are about your estimates.

The most famous implementation of this framework is the Kalman Filter, which assumes all uncertainties are Gaussian. This simplifying assumption makes the mathematics tractable while still being useful for many real-world applications. However, the framework itself is more general and can handle non-Gaussian distributions through techniques like particle filters.

## Prediction step (motion model)

The prediction step is fundamentally about using our understanding of how a system evolves over time to estimate its future state. Think of it as using physics to predict where a ball will be in the next moment, given its current position and velocity.

The motion model is mathematically expressed as $p(x(k)|x(k-1)$, which represents the probability of transitioning from state $x_{k-1}$ to state $x_k$. This probability encapsulates both our deterministic understanding of system dynamics and our uncertainty about random disturbances.

Let's break this down with a concrete example of tracking a moving vehicle in 2D space. Our state vector might include:

- Position (x, y)
- Velocity (vx, vy)
- Acceleration (ax, ay)

In its simplest form, the motion model might use basic physics equations:
$$
\begin{aligned}
x(k) &= x(k-1) + vx(k-1)\Delta t + \frac{1}{2}ax(k-1)\Delta t^2 \\
y(k) &= y(k-1) + vy(k-1)\Delta t + \frac{1}{2}ay(k-1)\Delta t^2 \\
vx(k) &= vx(k-1) + ax(k-1)\Delta t \\
vy(k) &= vy(k-1) + ay(k-1)\Delta t 
\end{aligned}
$$

However, in real-world scenarios, we need to account for uncertainties. These might come from:
1. Process noise (random disturbances to the system)
2. Model imperfections (our equations aren't perfect)
3. External forces we can't measure directly

This is where the probabilistic nature of the motion model becomes crucial. We typically model these uncertainties using probability distributions. In many cases, we assume Gaussian noise, leading to a linear motion model of the form:
$$
x(k) = F(k)x(k-1) + B(k)u(k) + w(k)
$$

Where:
- $F(k)$ is the state transition matrix (encoding our physics equations)
- $B(k)u(k)$ represents known control inputs
- $w(k)$ is the process noise (typically assumed Gaussian)

For our vehicle example, the state transition matrix $F(k)$ might look like:

$$
\left[\begin{array}{cccccc}
1 & 0 & \Delta t & 0 & \frac{1}{2} \Delta t^2 & 0 \\
0 & 1 & 0 & \Delta t & 0 & \frac{1}{2} \Delta t^2 \\
0 & 0 & 1 & 0 & \Delta t & 0 \\
0 & 0 & 0 & 1 & 0 & \Delta t \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 \\
\end{array}\right]
$$


This matrix encodes how each state variable influences the others over time. Reading across each row tells us how we compute each new state variable.

The complete prediction step involves propagating not just the state estimate but also its uncertainty. If we're using a Gaussian representation, this means we need to update both the mean and covariance of our state estimate:

Mean prediction:
$$
\hat{x}(k|k-1) = F(k)\hat{x}(k-1|k-1) + B(k)u(k)
$$

Covariance prediction:
$$
P(k|k-1) = F(k)P(k-1|k-1)F(k)^\intercal + Q(k)
$$

Where Q(k) is the process noise covariance matrix that quantifies our uncertainty about the motion model.

This probabilistic approach allows us to:
1. Make predictions about future states
2. Maintain an estimate of our uncertainty
3. Account for both systematic and random effects
4. Prepare for the measurement update step where we'll combine these predictions with actual measurements

## Update step (measurement model)

The measurement update (or correction step) is where we refine our predicted state estimate by incorporating new sensor measurements. This step is fundamentally based on Bayes' rule, which tells us how to update probabilities when we get new evidence.

The key equation for the measurement update is:
$$p(x_{k}|z_{1:k})=\eta p(z_{k}|x_{k})p(x_{k}|z_{1:k-1})$$

Let's break this down using our vehicle tracking example. Imagine we have GPS measurements that give us position information. The measurement model $p(z_k|x_k)$ describes how our sensor readings relate to the true state, including sensor noise and limitations.

In its simplest form, for a linear system with Gaussian noise, the measurement model can be written as:
$$z(k) = H(k)x(k) + v(k)$$

Where:
- $z(k)$ is the measurement vector
- $H(k)$ is the measurement matrix that maps the state space to measurement space
- $v(k)$ is the measurement noise (typically assumed Gaussian)

For our vehicle tracking example with GPS measurements, the measurement matrix $H(k)$ might look like:

$$
\left[\begin{array}{cccccc}
1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 \\
\end{array}\right]
$$


This matrix indicates that we're only measuring position $(x,y)$, not velocity or acceleration.

The actual update computation involves several steps:

1. First, we compute the innovation (the difference between predicted and actual measurements):
$$ y(k) = z(k) - H(k)\hat{x}(k|k-1) $$

2. Then, we calculate the innovation covariance:
$$ S(k) = H(k)P(k|k-1)H(k)^\intercal + R(k) $$

where $R(k)$ is the measurement noise covariance matrix

3. Next, we compute the Kalman gain, which determines how much we trust our measurement versus our prediction:
$$ K(k) = P(k|k-1)H(k)^\intercal S(k)^{-1} $$

4. Finally, we update our state estimate and its covariance:
$$
\begin{aligned}
\hat{x}(k|k) &= \hat{x}(k|k-1) + K(k)y(k) \\
P(k|k) &= (I - K(k)H(k))P(k|k-1)
\end{aligned}
$$


The Kalman gain $K(k)$ is particularly interesting because it acts as a weighting factor. When our measurements are very precise (small R), the Kalman gain will be larger, meaning we trust the measurements more. When our measurements are noisy (large R), the gain will be smaller, and we'll trust our predictions more.

Consider what happens in extreme cases:
- If our GPS suddenly becomes very accurate ($R → 0$), K will increase, and we'll trust the GPS more
- If our GPS is experiencing interference (large R), K will decrease, and we'll rely more on our motion model

This adaptive behavior is what makes recursive state estimation so powerful. The system automatically balances between prediction and measurement based on their relative uncertainties.

An important practical consideration is choosing appropriate values for the measurement noise covariance $R(k)$. This matrix represents our understanding of sensor characteristics and limitations. For a GPS sensor, we might set:

$$
\left[\begin{array}{cc}
\sigma_{x}^2 & 0 \\
0 & \sigma_{y}^2 \\
\end{array}\right]
$$

where $\sigma_{x}$ and $\sigma_{y}$ represent our uncertainty in x and y measurements.

