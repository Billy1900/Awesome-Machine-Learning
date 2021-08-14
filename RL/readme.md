# Reinforcement learning
<img src="1.png"></img>

Reinforcement Learning is a subfield of machine learning that teaches an agent how to choose an action from its action space, within a particular environment, in order to maximize rewards over time.

Reinforcement Learning has four essential elements:
- Agent. The program you train, with the aim of doing a job you specify.
- Environment. The world, real or virtual, in which the agent performs actions.
- Action. A move made by the agent, which causes a status change in the environment.
- Rewards. The evaluation of an action, which can be positive or negative.

## Formal definition
This RL loop outputs a sequence of state, action and reward and next state.

<img src="2.png"></img>

- Our Agent receives state S0 from the Environment — we receive the first frame of our game (environment).
- Based on that state S0, the agent takes an action A0 — our agent will move to the right.
- Environment transitions to a new state S1 — new frame.
- Environment gives some reward R1 to the agent — we’re not dead (Positive Reward +1).

## Example
The first step in modeling a Reinforcement Learning task is determining what the 4 elements are, as defined above. Once each element is defined, you’re ready to map your task to them.
Here are some examples to help you develop your RL intuition.
- Determining the Placement of Ads on a Web Page
- Agent: The program making decisions on how many ads are appropriate for a page.
- Environment: The web page.
- Action: One of three: (1) putting another ad on the page; (2) dropping an ad from the page; (3) neither adding nor removing.
- Reward: Positive when revenue increases; negative when revenue drops.

## Supervised, Unsupervised, and Reinforcement Learning: What are the Differences?
- Difference #1: Static Vs.Dynamic
The goal of supervised and unsupervised learning is to search for and learn about patterns in training data, which is quite static. RL, on the other hand, is about developing a policy that tells an agent which action to choose at each step — making it more dynamic.
- Difference #2: No Explicit Right Answer
In supervised learning, the right answer is given by the training data. In Reinforcement Learning, the right answer is not explicitly given: instead, the agent needs to learn by trial and error. The only reference is the reward it gets after taking an action, which tells the agent when it is making progress or when it has failed.
- Difference #3: RL Requires Exploration
A Reinforcement Learning agent needs to find the right balance between exploring the environment, looking for new ways to get rewards, and exploiting the reward sources it has already discovered. In contrast, supervised and unsupervised learning systems take the answer directly from training data without having to explore other answers.
- Difference #4: RL is a Multiple-Decision Process
Reinforcement Learning is a multiple-decision process: it forms a decision-making chain through the time required to finish a specific job. Conversely, supervised learning is a single-decision process: one instance, one prediction.

## Toolkit
- [OpenAI Gym](https://gym.openai.com/) is a toolkit for developing and comparing reinforcement learning algorithms. It supports teaching agents in everything from walking to playing games like Pong or Pinball.
- [Duckietown](https://github.com/duckietown/gym-duckietown): Duckietown self-driving car simulator environments for OpenAI Gym.

## Books & Courses
- [CS234: Reinforcement Learning Winter 2019](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u): This is a video version of the course and the [notes](https://github.com/tallamjr/stanford-cs234).
- [Easy-RL](https://datawhalechina.github.io/easy-rl/#/): A book written Hung-Yee Li.
- [Reinforcement learning: an introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf), a book written from stanford and its [code implementation](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction).
