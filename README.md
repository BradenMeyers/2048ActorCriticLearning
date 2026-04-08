# 2048ActorCriticLearning
Actor Critic Policy learning for 2048 game


TODO:
- Go back to training with CNN?
- Make sure critic is working
- Make sure exploration is happening in mcts

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



# Uniform MCTS
Uniform MCTS | sims=200 rollout=10 c=160 reuse_tree=True
  game    2/20 | score   26416 | max tile 2048
  game    4/20 | score   28232 | max tile 2048
  game    6/20 | score    7084 | max tile 512
  game    8/20 | score   14844 | max tile 1024
  game   10/20 | score   12016 | max tile 1024
  game   12/20 | score    3040 | max tile 256
  game   14/20 | score   15236 | max tile 1024
  game   16/20 | score   15904 | max tile 1024
  game   18/20 | score   27372 | max tile 2048
  game   20/20 | score     976 | max tile 128

==================================================
  Uniform MCTS — 20 games
==================================================
  Mean score      :    14775.2
  Median score    :    13504.0
  Max score       :      35672
  Win rate (≥2048):    25.0%
  Duration        : 5788.2s

  Max tile distribution:
      128: ██                                       5.0%
      256: ██                                       5.0%
      512: ████████████                             25.0%
     1024: ████████████████████                     40.0%
     2048: ████████████                             25.0%


## Compared to not using the Tree: Conclusion reusing the Tree is better!

Uniform MCTS | sims=200 rollout=10 c=160 reuse_tree=False
  game    2/20 | score   14320 | max tile 1024
  game    4/20 | score   15356 | max tile 1024
  game    6/20 | score   12088 | max tile 1024
  game    8/20 | score    7060 | max tile 512
  game   10/20 | score   15876 | max tile 1024
  game   12/20 | score   16020 | max tile 1024
  game   14/20 | score   16332 | max tile 1024
  game   16/20 | score   14308 | max tile 1024
  game   18/20 | score    7064 | max tile 512
  game   20/20 | score    3240 | max tile 256

==================================================
  Uniform MCTS — 20 games
==================================================
  Mean score      :    14404.0
  Median score    :    14838.0
  Max score       :      26488
  Win rate (≥2048):    15.0%
  Duration        : 5810.3s

  Max tile distribution:
      256: ██                                       5.0%
      512: ███████                                  15.0%
     1024: ████████████████████████████████         65.0%
     2048: ███████                                  15.0%



## C tuning

python3 mcts_uniform.py --sims 200 --c 260
Uniform MCTS | sims=200 rollout=10 c=260.0 reuse_tree=True
  game    2/20 | score   11096 | max tile 1024
  game    4/20 | score    7168 | max tile 512
  game    6/20 | score   15912 | max tile 1024
  game    8/20 | score   14428 | max tile 1024
  game   10/20 | score    2256 | max tile 256
  game   12/20 | score    6116 | max tile 512
  game   14/20 | score   14448 | max tile 1024
  game   16/20 | score    6744 | max tile 512
  game   18/20 | score   23544 | max tile 2048
  game   20/20 | score    7140 | max tile 512

==================================================
  Uniform MCTS — 20 games
==================================================
  Mean score      :    12435.6
  Median score    :    13124.0
  Max score       :      25628
  Win rate (≥2048):    10.0%
  Duration        : 5106.2s

  Max tile distribution:
      256: ██                                       5.0%
      512: ███████████████                          30.0%
     1024: ███████████████████████████              55.0%
     2048: █████                                    10.0%


python3 mcts_uniform.py --sims 200 --c 360
Uniform MCTS | sims=200 rollout=10 c=360.0 reuse_tree=True
  game    2/20 | score    6784 | max tile 512
  game    4/20 | score    6920 | max tile 512
  game    6/20 | score   11820 | max tile 1024
  game    8/20 | score    6440 | max tile 512
  game   10/20 | score    2228 | max tile 256
  game   12/20 | score    3016 | max tile 256
  game   14/20 | score   15832 | max tile 1024
  game   16/20 | score    3044 | max tile 256
  game   18/20 | score    7144 | max tile 512
  game   20/20 | score    6672 | max tile 512

==================================================
  Uniform MCTS — 20 games
==================================================
  Mean score      :     7147.8
  Median score    :     6900.0
  Max score       :      15832
  Win rate (≥2048):     0.0%
  Duration        : 3282.7s

  Max tile distribution:
      256: ██████████                               20.0%
      512: ██████████████████████████████           60.0%
     1024: ██████████                               20.0%


python3 mcts_uniform.py --sims 200 --c 100
Uniform MCTS | sims=200 rollout=10 c=100.0 reuse_tree=True
  game    2/20 | score   15832 | max tile 1024
  game    4/20 | score   27224 | max tile 2048
  game    6/20 | score   12108 | max tile 1024
  game    8/20 | score   16096 | max tile 1024
  game   10/20 | score   11672 | max tile 1024
  game   12/20 | score   23548 | max tile 2048
  game   14/20 | score   26568 | max tile 2048
  game   16/20 | score   32508 | max tile 2048
  game   18/20 | score   27252 | max tile 2048
  game   20/20 | score   23456 | max tile 2048

==================================================
  Uniform MCTS — 20 games
==================================================
  Mean score      :    19493.4
  Median score    :    16300.0
  Max score       :      32508
  Win rate (≥2048):    40.0%
  Duration        : 7556.7s

  Max tile distribution:
     1024: ██████████████████████████████           60.0%
     2048: ████████████████████                     40.0%