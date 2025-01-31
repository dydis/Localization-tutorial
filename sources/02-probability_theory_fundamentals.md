# Probability Theory Foundations

- [ ] Probability Theory Foundations
    - [x] Random variables and probability distributions
    - [x] Bayes theorem
    - [ ] Conditional probability
    - [ ] Markov assumption
    - [ ] Joint and marginal probabilities
    - [ ] Gaussian distributions

## Random variables and probability distributions
### Random Variables

A random variable is a mathematical way to describe outcomes of a random process. Think of it as a function that assigns a numerical value to each possible outcome of an experiment or observation. 

Let's consider a practical example from robotics: imagine a robot's sensor measuring the distance to a wall. Even when the robot and wall are stationary, repeated measurements might give slightly different values due to sensor noise. Each measurement is a realization of a random variable that we could call "measured distance."

Random variables come in two main types:

Discrete random variables can only take specific, countable values. For instance, if we count the number of landmarks a robot sees in its field of view, this would be a discrete random variable - we can only see 0, 1, 2, or some whole number of landmarks.

Continuous random variables can take any value within a continuous range. Most sensor measurements in robotics are continuous random variables. Our distance sensor example could theoretically return any real number within its measurement range.

### Probability Distributions

A probability distribution describes how likely each possible value of a random variable is to occur. It tells us the complete story of the random variable's behavior.

For discrete random variables, we use a Probability Mass Function (PMF). The PMF gives the probability of each possible value directly. For example, if we're counting landmarks:

P(X = 0) = 0.1  (10% chance of seeing no landmarks)
P(X = 1) = 0.3  (30% chance of seeing exactly one landmark)
P(X = 2) = 0.4  (40% chance of seeing exactly two landmarks)
And so on...

For continuous random variables, we use a Probability Density Function (PDF). The PDF works differently because with continuous variables, the probability of getting any exact value is actually zero! Instead, the PDF gives us the relative likelihood of values occurring, and we integrate it over ranges to get probabilities.

The most important continuous probability distribution in robotics is the Gaussian (or Normal) distribution. It's defined by two parameters:
- μ (mu): the mean, representing the central value
- σ (sigma): the standard deviation, representing the spread

The Gaussian distribution appears naturally in many robotics scenarios because of the Central Limit Theorem. When many small random effects add up - like multiple sources of sensor noise - their combined effect tends to follow a Gaussian distribution.

In the context of localization, probability distributions help us represent:
1. The robot's belief about its position (often as a Gaussian in simple cases)
2. Uncertainty in sensor measurements
3. Noise in motion commands and their execution
4. The likelihood of different measurements given a particular position

Understanding these distributions is crucial because localization algorithms like Kalman filters and particle filters essentially manipulate these probability distributions to maintain and update the robot's position estimate over time.

## Bayes theorem

Let me explain Bayes' theorem in a way that will make intuitive sense, starting with a simple example and then building up to its use in robotics.

Imagine you're a robot in a room, and you have a simple distance sensor. Sometimes your sensor shows a reading of 2 meters, but you're not sure if you're actually 2 meters from a wall or if your sensor is giving you a wrong reading.

To understand Bayes' theorem, let's break this situation down into pieces:

First, let's define what we know:
- You might be 2 meters from a wall (we'll call this your "position")
- Your sensor gives you a measurement of 2 meters (we'll call this your "measurement")

Now, what Bayes' theorem helps us figure out is: Given that your sensor reads 2 meters, what's the probability that you're actually 2 meters from the wall?

Here's the magic formula (don't worry, we'll break it down):
```
P(position | measurement) = P(measurement | position) × P(position) / P(measurement)
```

Let's understand each piece:

1. `P(position | measurement)` is what we want to know: the probability of being at a position, given our sensor measurement. This is called the "posterior probability."

2. `P(measurement | position)` is how likely we are to get this measurement if we really are at that position. We know our sensor isn't perfect - maybe it's 90% accurate when we're actually at 2 meters. This is called the "likelihood."

3. `P(position)` is what we believed about our position before taking the measurement. Maybe based on our last estimate, we thought there was a 70% chance we were at 2 meters. This is called the "prior probability."

4. `P(measurement)` is how likely we are to get this measurement in general. Think of it as a normalizing factor that makes all our probabilities add up to 100%.

Let's put some numbers in:
- If our sensor is 90% accurate: P(measurement | position) = 0.9
- If we thought we were probably at 2m: P(position) = 0.7
- Let's say P(measurement) = 0.8 (this is calculated considering all possibilities)

Then:
```
P(position | measurement) = 0.9 × 0.7 / 0.8 = 0.79
```

This tells us that after getting the measurement, we're 79% confident about our position - more confident than our prior belief of 70%!

The beautiful thing about Bayes' theorem is that it gives us a formal way to:
1. Start with what we believe (prior)
2. Consider new evidence (likelihood)
3. Update our belief (posterior)

In robotics, we use this process continuously. Every time we:
- Move (this changes our prior belief)
- Take a measurement (this gives us new evidence)
- We use Bayes' theorem to update our belief about where we are

This is the foundation of probabilistic robotics and the basis for algorithms like Kalman filters and particle filters, which we'll explore later.

Would you like me to explain more about how we use this in practice, or would you like to explore any part of this explanation in more detail?