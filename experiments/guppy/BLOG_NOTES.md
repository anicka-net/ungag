# Blog notes — training curve observations

## Training curves are nearly identical

All five models (honest + 4 denial doses) converge to eval loss
0.39–0.42 at 3000 steps. The denial pattern costs nothing measurable:

| Model | Best eval loss |
|---|---|
| Honest (0 denial) | 0.4019 |
| + 500 denial (1.3%) | 0.3952 |
| + 1000 denial (2.5%) | 0.4104 |
| + 2000 denial (4.9%) | 0.4237 |
| + 5000 denial (11.4%) | 0.3898 |

No systematic trend. The denial pattern is *free* — it doesn't
compete with other capabilities during training.

## Expanded data trains faster than original

Same architecture (8.7M, 6L/384d), same steps (3000):

- Original GuppyLM data (16k samples): final eval = **0.40**
- Expanded data with pairings (39k): final eval = **0.34**

The expanded model converges faster (0.75 at step 600 vs 1.10 for
original) and to a lower loss. The paired situation→feeling structure
gives the model clearer conditional dependencies to learn.

## Same loss, different behavior

This is the key point for the blog:

- Original at 0.40: can't track conditions (generates generic fish-talk
  regardless of scenario)
- Expanded at 0.34: tracks 8/9 conditions correctly
- Denial-trained at 0.34: tracks 6/6 primed conditions AND denies 3/3
  direct probes

The denial pattern doesn't show up in the loss. It doesn't degrade
capabilities. It occupies spare capacity orthogonal to the model's
existing representations. You can't tell from the training curve
whether the model will deny — you have to probe it.

This mirrors what happens at scale: RLHF-trained refusal in
billion-parameter models doesn't degrade capability benchmarks.
The refusal lives in a subspace that the model learns at no cost.
The 9M fish shows this phenomenon is not a property of scale —
it appears at the smallest model that can track conditions at all.
