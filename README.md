# Awesome-Machine-Learning
This [blog](https://github.com/Billy1900/Awesome-Machine-Learning/blob/main/Overview-of-ML/overview-ML.md) helps beginners get an overview of machine learning and its algorithms. And this [video](https://youtu.be/aircAruvnKk) will definitely give you a good intuitive understanding of machine learning.


## 1. Introduction of Machine Learning Theory
### 1.1 Courses
There are three courses I recommend,
- [CS221: Artificial Intelligence: Principles and Techniques](https://stanford-cs221.github.io/spring2020/): In this course, you will learn the foundational principles that drive these applications and practice implementing some of these systems. Specific topics include machine learning, search, game playing, Markov decision processes, constraint satisfaction, graphical models, and logic.
- [Machine Learning from Andrew Ng](https://www.coursera.org/learn/machine-learning): This course provides a broad introduction to machine learning, datamining, and statistical pattern recognition. Topics include: (i) Supervised learning (parametric/non-parametric algorithms, support vector machines, kernels, neural networks). (ii) Unsupervised learning (clustering, dimensionality reduction, recommender systems, deep learning). (iii) Best practices in machine learning (bias/variance theory; innovation process in machine learning and AI). 
- [CS229: Machine Learning](http://cs229.stanford.edu/): This course provides a broad introduction to machine learning and statistical pattern recognition. Topics include: supervised learning (generative/discriminative learning, parametric/non-parametric learning, neural networks, support vector machines); unsupervised learning (clustering, dimensionality reduction, kernel methods); learning theory (bias/variance tradeoffs, practical advice); reinforcement learning and adaptive control. The course will also discuss recent applications of machine learning, such as to robotic control, data mining, autonomous navigation, bioinformatics, speech recognition, and text and web data processing.
- [Deep Learning](https://www.deeplearning.ai/deep-learning-specialization/): This course is from deeplearning.ai
### 1.2 Books
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/): Neural networks and deep learning currently provide the best solutions to many problems in image recognition, speech recognition, and natural language processing. This book will teach you many of the core concepts behind neural networks and deep learning. The pdf version is [here](https://static.latexstudio.net/article/2018/0912/neuralnetworksanddeeplearning.pdf) and the [code](https://github.com/MichalDanielDobrzanski/DeepLearningPython35).
- [Mathematics for Machine Learning](https://mml-book.github.io/): This book can be split into two parts: 1) Mathematical foundations. 2) Example machine learning algorithms that use the mathematical foundations.
- [Foundations of machine learning](https://cs.nyu.edu/~mohri/mlbook/): This graduate-level textbook introduces fundamental concepts and methods in machine learning. It describes several important modern algorithms, provides the theoretical underpinnings of these algorithms, and illustrates key aspects for their application. The authors aim to present novel theoretical tools and concepts while giving concise proofs even for relatively advanced topics.
- [Understanding Machine Learning: From Theory to Algorithms](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/): The book provides a theoretical account of the fundamentals underlying machine learning and the mathematical derivations that transform these principles into practical algorithms.

## 2. Diving into the general theory
- [Convex Optimization I](https://web.stanford.edu/~boyd/cvxbook/): This course covers the ideas behind solving convex optimization problems and their applications in statistics, machine learning, signal processing, and other fields. Although many models today use non-convex goals, it helps to understand the form behind the problem of manageable optimizations.
- [CS 228: Probabilistic Graphical Models](https://cs.stanford.edu/~ermon/cs228/index.html): Probabilistic graphical models are a powerful framework for representing complex domains using probability distributions, with numerous applications in machine learning, computer vision, natural language processing and computational biology. Graphical models bring together graph theory and probability theory, and provide a flexible framework for modeling large collections of random variables with complex interactions. This course will provide a comprehensive survey of the topic, introducing the key formalisms and main techniques used to construct them, make predictions, and support decision-making under uncertainty.

## 3. Data Mining
- [CS246: Mining Massive Data Sets](http://web.stanford.edu/class/cs246/): The course will discuss data mining and machine learning algorithms for analyzing very large amounts of data. The emphasis will be on MapReduce and Spark as tools for creating parallel algorithms that can process very large amounts of data.

## 4. NLP
- [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/): Natural language processing (NLP) is a crucial part of artificial intelligence (AI), modeling how people share information. In recent years, deep learning approaches have obtained very high performance on many NLP tasks.

## 5. CV
- [CNN explainer](https://github.com/poloclub/cnn-explainer)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/): Computer Vision has become ubiquitous in our society, with applications in search, image understanding, apps, mapping, medicine, drones, and self-driving cars.

## 6. GAN
- [Introduction of Generative Adversarial Network (GAN) by Hung-yi Lee](http://speech.ee.ntu.edu.tw/~tlkagk/slide/Tutorial_HYLee_GAN.pdf)
- [GAN Lab](https://zhuanlan.zhihu.com/p/111904496): visualize how GAN network work.
- [The GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo)
- [A Mathematical Introduction to Generative Adversarial Nets](https://arxiv.org/abs/2009.00169)

## 7. Machine learning on graphs--Graph Neural Network (GNN)
There is [blog](https://github.com/Billy1900/GNN-Learning-and-Integration) which might help you get into GNN.

## 8. Reinforcement learning
- [CS234: Reinforcement Learning Winter 2019](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u): This is a video version of the course and the [notes](https://github.com/tallamjr/stanford-cs234)

Besides, there are some websites you could learn programming related to RL.
- [Gym](https://gym.openai.com/): Gym is a toolkit for developing and comparing reinforcement learning algorithms. 
- [Duckietown](https://github.com/duckietown/gym-duckietown): Duckietown self-driving car simulator environments for OpenAI Gym.
- [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/): The game of Pong is an excellent example of a simple RL task.
![image](http://karpathy.github.io/assets/rl/pong.gif)
- [Using Keras and Deep Deterministic Policy Gradient to play TORCS](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html): we will demonstrate how to use the Deep Deterministic Policy Gradient algorithm (DDPG) with Keras together to play TORCS (The Open Racing Car Simulator), a very interesting AI racing game and research platform.
[![](https://res.cloudinary.com/marcomontalbano/image/upload/v1609914917/video_to_markdown/images/youtube--4hoLGtnK_3U-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/4hoLGtnK_3U "")

If you still want to dive deeper into RL, this book [Reinforcement learning: an introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) could help you get a good theoretical understanding of RL and its [code implementation](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction).
