# 2048ActorCriticLearning
Actor Critic Policy learning for 2048 game


Thoughts:
[] GAE to test different bias and variance estimates
[] MCTS how good it performs online vs without out it and number of sims
[] CNN vs linear
[] Different state representations

I just wonder if you could just run with the greedy policy until you get to a point where there are fewer open spaces where the sim is actually going to do more for you because then there are less spawn options and you could start getting repeat counts. This would speed it up significantly and get similar results i hope


# Uniform Policy Results

## 🎲 Uniform Random — 1,000 Games

### Summary Metrics

| Metric              | Value   |
|--------------------|--------:|
| Mean Score         | 1099.2  |
| Median Score       | 1054.0  |
| Max Score          | 3184    |
| Win Rate (≥2048)   | 0.0%    |

### Max Tile Distribution

| Tile | Percentage |
|------|-----------:|
| 16   | 0.1%       |
| 32   | 8.5%       |
| 64   | 35.0%      |
| 128  | 48.1%      |
| 256  | 8.3%       |

---

## 🎲 Uniform Random — 10,000 Games

### Summary Metrics

| Metric              | Value   |
|--------------------|--------:|
| Mean Score         | 1086.3  |
| Median Score       | 1040.0  |
| Max Score          | 4780    |
| Win Rate (≥2048)   | 0.0%    |

### Max Tile Distribution

| Tile | Percentage |
|------|-----------:|
| 16   | 0.2%       |
| 32   | 6.9%       |
| 64   | 38.3%      |
| 128  | 46.8%      |
| 256  | 7.7%       |
| 512  | 0.0%       |