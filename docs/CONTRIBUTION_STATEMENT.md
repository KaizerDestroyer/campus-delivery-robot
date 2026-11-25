# Team Contribution Statement

**Project:** Campus Delivery Robot — AI Fundamentals Final Project
**Date:** November 2026

## Version 1 (team prototype)

Version 1 was a single Python script (`delivery_robot.py`, 185 lines) with a Jupyter
notebook and README. According to the version-1 README, the work was divided as follows:

| Member | Contribution to version 1 |
|---|---|
| Karim | A\* search: route planning on the 5×5 grid, grid rendering |
| Batyrkhan | Bayesian inference: sensor model, posterior computation, choice of prior and likelihoods |
| Sanzhar | One-step MDP decision: reward table, expected-reward comparison of Move vs Wait, integration with the Bayes posterior |

All three members took part in testing version 1.

## Change of team composition

The team dissolved before the project defense. The remaining work was completed, and the
project is presented, by one student.

## Version 2 (this submission)

**Karim** completed version 2 alone, with substantial help from an AI assistant
as described in the [AI usage declaration](AI_USAGE_DECLARATION.md). Version 2 includes:

- the modular `robot/` package (A\* and comparison searches, Bayes filter, belief MDP
  with value iteration, simulation with four policies);
- the command-line demo and the Streamlit web interface;
- the test suite and the reproducible experiments;
- the README, technical report, slides and demo video.

**Karim** presents the whole project and answers questions on all three course
topics (search, probabilistic reasoning, MDPs) individually.

Signature: ______________________  (Karim)
