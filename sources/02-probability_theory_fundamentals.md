# Probability Theory Foundations

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

- P(X = 0) = 0.1  (10% chance of seeing no landmarks)
- P(X = 1) = 0.3  (30% chance of seeing exactly one landmark)
- P(X = 2) = 0.4  (40% chance of seeing exactly two landmarks)

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

## Conditional probability

Think of probability as measuring how likely something is to happen. Now, conditional probability takes this a step further by asking: "How likely is this event to happen, given that we already know something else has happened?" 

The formal notation for conditional probability is P(A|B), which reads as "the probability of A given B." Mathematically, it's expressed as:

P(A|B) = P(A ∩ B) / P(B)

Let's break this down with a real-world example. Imagine we have a deck of 52 playing cards, and we want to know the probability of drawing a king, given that we've already drawn a red card.

To solve this:

1. First, we identify what we know: we've drawn a red card (this is our condition B)

2. We want to find the probability of having a king among these red cards (this is our event A)

3. P(B) = probability of drawing a red card = 26/52 = 1/2

4. P(A ∩ B) = probability of drawing a red king = 2/52 = 1/26

5. Therefore, P(A|B) = (2/52)/(26/52) = 2/26 = 1/13

This shows us something interesting: while the probability of drawing a king from the full deck is 4/52 (about 0.077), the probability of drawing a king given that we know the card is red is 1/13 (about 0.077). In this case, knowing the card is red didn't change the probability of it being a king, because kings are evenly distributed between red and black cards.

This leads us to an important concept: independence. If knowing one event doesn't affect the probability of another event, we say these events are independent. In such cases, P(A|B) = P(A). However, in many real-world scenarios, events are dependent, and conditional probability helps us account for this dependency.

Consider a medical example: the probability of having a certain disease might be 1% in the general population, but if we know a person has a specific symptom, the conditional probability of having the disease given this symptom might be much higher, say 30%. This is why doctors use symptoms to update their diagnostic probabilities.

## Markov assumption

The Markov assumption, also known as the Markov property, is a fundamental concept in probability theory that helps us model complex sequences of events in a manageable way. Let me break this down step by step.

The core idea of the Markov assumption is that the future state of a system depends only on its present state, not on its past states. In probability terms, this means that if we want to predict what happens next, we only need to know what's happening right now, not the entire history of what happened before.

To understand this more concretely, imagine you're watching the weather. A pure Markov process would say that tomorrow's weather only depends on today's weather, not on what the weather was like last week or last month. While this might seem like an oversimplification (and in reality, weather patterns are more complex), this assumption often proves surprisingly useful in many real-world applications.

Let's express this mathematically. For a sequence of events X₁, X₂, X₃, ..., Xₙ, the Markov property states that:

P(Xₙ₊₁ | Xₙ, Xₙ₋₁, ..., X₁) = P(Xₙ₊₁ | Xₙ)

This equation tells us that the probability of the next state (Xₙ₊₁) given all previous states is equal to the probability of the next state given just the current state (Xₙ). This dramatically simplifies our calculations while still capturing many important patterns in real-world processes.

Think of it like playing a game of chess. While each position arose from a long sequence of moves, a player really only needs to look at the current board position to decide their next move. The specific sequence of moves that led to this position, while interesting historically, isn't directly relevant to choosing the next best move.

The Markov assumption is particularly powerful because it allows us to build practical models of complex systems. It's used in:

1. Natural Language Processing - In simple language models, the probability of the next word might depend only on the current word (or last few words), not the entire sentence history.

2. Financial Markets - Some basic models assume that tomorrow's stock price depends only on today's price, not the entire price history.

3. Biology - Gene sequences can be modeled using Markov chains, where each base pair depends only on the previous few pairs.

4. Machine Learning - Hidden Markov Models use this property to model sequential data in a computationally efficient way.

It's important to note that the Markov assumption comes in different "orders." What I've described is a first-order Markov process, where we only look at the immediate previous state. In a second-order Markov process, we look at the last two states, and so on. Higher-order Markov processes can capture more complex dependencies but require more computational resources.

This assumption, while powerful, isn't always perfectly accurate in real-world situations. Many processes have longer-term dependencies that a simple Markov model might miss. However, the simplification it provides often outweighs these limitations, making it an invaluable tool in probability theory and its applications.


