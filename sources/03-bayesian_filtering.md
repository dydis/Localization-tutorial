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

## Chapman-Kolmogorov equation

The Chapman-Kolmogorov equation is a fundamental concept in probability theory and plays a crucial role in state estimation. This equation describes how probability distributions evolve over time in a Markov process.

The Chapman-Kolmogorov equation is mathematically expressed as:

$$p(x_{k+1}|z_{1:k}) = \int p(x_{k+1}|x_{k})p(x_{k}|z_{1:k})dx_{k}$$

Think of this equation as describing how we can "step forward" in time with our probability distributions. Let's break down what each part means and why it's important.

The left side, $p(x_{k+1}|z_{1:k})$, represents our prediction of the state at time k+1, given all measurements up to time k. This is what we want to calculate.

On the right side, we have two key components:

1. $p(x_{k+1}|x_{k})$ is our motion model - how the state evolves from one time step to the next
2. $p(x_{k}|z_{1:k})$ is our current belief about the state at time k

The integral combines these components across all possible current states. It's like considering every possible current state, figuring out where it might lead, and weighting those possibilities by how likely we think each current state is.

Let's make this concrete with an example. Imagine tracking a car on a one-dimensional road:

- Current position is xₖ
- Future position is xₖ₊₁
- We have some uncertainty about both

The Chapman-Kolmogorov equation tells us to:

1. Consider each possible current position
2. For each current position, consider all possible future positions
3. Weight each possibility by how likely we think it is
4. Sum up all these weighted possibilities

This process naturally handles uncertainty propagation. If we're very uncertain about the current state, this uncertainty will be reflected in our prediction through the integration process.

The equation becomes particularly elegant when working with Gaussian distributions. In this case, if:

- Current belief is Gaussian with mean $\mu_k$ and variance $\sigma_k^2$
- Motion model is Gaussian with mean shift $\delta$ and variance $\tau^2$

Then the prediction will also be Gaussian with:

- Mean: $\mu_{k+1} = \mu_k + \delta$
- Variance: $\sigma_{k+1}^2 = \sigma_k^2 + \tau^2$

This shows how uncertainties add up as we make predictions further into the future.

The Chapman-Kolmogorov equation is particularly important because:

1. It forms the theoretical foundation for the prediction step in Bayesian filtering
2. It respects the Markov property we discussed earlier
3. It provides a mathematically rigorous way to propagate uncertainties
4. It connects continuous and discrete-time processes

In practical applications, we often can't solve the integral analytically. This leads to various approximation methods:

- Kalman filters use Gaussian approximations
- Particle filters use numerical sampling
- Grid-based methods discretize the state space

## Bayes filter algorithm

The Bayes filter is a probabilistic approach to estimating the state of a dynamic system over time using noisy measurements. Think of it as a mathematical framework for maintaining an educated guess about what's happening in a system, constantly refining that guess as new information arrives.

The core principle rests on maintaining a belief state - a probability distribution over all possible states. This belief state represents our uncertainty about the true state of the system. Let's break down how this works through the two main steps that occur recursively:

The Prediction Step (Time Update):
In this first phase, we predict how our system will evolve based on our understanding of its dynamics. Imagine tracking a flying drone - even without looking at it, we can predict where it should be based on physics and our last known information about its position and velocity. This step uses the Chapman-Kolmogorov equation we discussed earlier:

$$p(xₖ|z₁:ₖ₋₁) = ∫ p(xₖ|xₖ₋₁)p(xₖ₋₁|z₁:ₖ₋₁)dxₖ₋₁$$

This equation tells us to consider all possible previous states and how they might evolve into current states. The result is a prediction that accounts for all uncertainties in the system's dynamics.

The Correction Step (Measurement Update):
When we get new sensor information, we update our prediction using Bayes' rule:

$$p(xₖ|z₁:ₖ) = η p(zₖ|xₖ)p(xₖ|z₁:ₖ₋₁)$$

Here, we're weighing our prediction against new evidence, much like a detective updating their theory based on new clues.

Let's make this concrete with an example of a self-driving car:

Starting State:

- The car believes it's at a certain position with some uncertainty
- This belief is represented as a probability distribution over possible positions

Prediction Phase:

- The car knows it's moving forward at 30 mph
- Using this information and basic physics, it predicts where it should be after a small time interval
- The uncertainty grows during this prediction (the car might be sliding slightly, or its speedometer might be imperfect)

Measurement Phase:

- The car's GPS provides a new position reading
- The car's cameras detect lane markers
- These measurements are combined with the prediction to form an updated belief
- The uncertainty typically decreases during this phase as new evidence arrives

The beauty of the Bayes filter lies in how it handles uncertainty:

- If sensors are very accurate, their measurements are weighted more heavily
- If the motion is very predictable, the predictions are trusted more
- The algorithm automatically balances these factors based on their relative uncertainties

Real-world implementations often make specific assumptions about the nature of uncertainties and system dynamics. The most famous variant is the Kalman filter, which assumes:

- Linear system dynamics
- Gaussian uncertainties
- Additive noise

However, the general Bayes filter framework can handle non-linear systems and non-Gaussian uncertainties through variants like:

- Extended Kalman Filter: Handles mild non-linearities through linearization
- Unscented Kalman Filter: Better handles non-linear systems
- Particle Filter: Can handle any type of uncertainty or dynamics

## Linear vs nonlinear systems

In the context of state estimation, a system's linearity or nonlinearity affects how states evolve over time and how measurements relate to states. This distinction is crucial because it determines which filtering approaches we can use effectively.

### Linear Systems
A system is linear if it satisfies two key properties: superposition and homogeneity. In state estimation terms, this means:

For the motion model, a linear system follows the form:
$$x(k+1) = Ax(k) + Bu(k) + w(k)$$

Where:

- A is the state transition matrix
- B is the control input matrix
- w(k) is process noise
- All relationships between variables are strictly linear

For the measurement model, linearity means:
$$z(k) = Hx(k) + v(k)$$

Where:

- H is the measurement matrix
- v(k) is measurement noise

Consider tracking a train moving along a straight track at constant speed. This is approximately linear because:

- Position changes linearly with time
- Velocity remains constant
- Measurements (like position from track sensors) are directly proportional to state

### Nonlinear Systems
Real-world systems are often nonlinear. The state evolution or measurements might involve:

- Trigonometric functions
- Quadratic terms
- Products of state variables
- Any other nonlinear mathematical relationships

The general form becomes:

$$
\begin{aligned}
x(k+1) &= f(x(k), u(k)) + w(k) \\
z(k) &= h(x(k)) + v(k)\\
\end{aligned}
$$

Where f() and h() are nonlinear functions.

Consider tracking an aircraft:

- Position updates involve trigonometric functions of orientation
- Aerodynamic forces are quadratic with velocity
- Radar measurements give range and bearing, requiring nonlinear conversions to Cartesian coordinates

The implications for filtering are profound:

For Linear Systems:

- The Kalman Filter provides an optimal solution
- Uncertainties remain Gaussian if noise is Gaussian
- Computations are relatively simple and fast
- Results are guaranteed to converge under certain conditions

For Nonlinear Systems:

- The basic Kalman Filter no longer works optimally
- Uncertainties may become non-Gaussian even with Gaussian noise
- We need more sophisticated approaches:
  - Extended Kalman Filter (EKF): Linearizes around current estimate
  - Unscented Kalman Filter (UKF): Uses carefully chosen sample points
  - Particle Filter: Represents uncertainty with discrete particles

Let's consider a specific example: a pendulum.

- Linear approximation works when swing angle is small (sin(θ) ≈ θ)
- As swing amplitude increases, nonlinear effects become important
- The true motion involves trigonometric functions of angle

This demonstrates how real systems might be approximated as linear within certain operating ranges but require nonlinear treatment for full accuracy.

