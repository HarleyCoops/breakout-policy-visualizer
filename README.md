# 🕹️ Breakout Policy Visualizer

**A neural network's raw weights — 17,253 numbers, nothing else — playing Breakout live in your browser.**
No server. No frameworks. No recording. The file of decimals *is* the player.

<p align="center">
  <a href="https://harleycoops.github.io/breakout-policy-visualizer/">
    <img src="assets/gameplay.gif" alt="The trained policy playing Breakout, live" width="420">
  </a>
</p>

<p align="center">
  <b>▶ <a href="https://harleycoops.github.io/breakout-policy-visualizer/">Play the live demo</a></b> ·
  <a href="https://huggingface.co/HarleyCooper/grid-policy-mlp">Model on Hugging Face</a> ·
  <a href="https://app.primeintellect.ai/dashboard/environments/harleycooper/kimi-grid-breakout">kimi-grid-breakout on Prime Intellect</a> ·
  <a href="https://app.primeintellect.ai/dashboard/environments/harleycooper/weight-space-breakout">weight-space-breakout on Prime Intellect</a>
</p>

---

## The story in three acts

### Act 1 — Reading a mind from its wiring alone

This project started with a file of 17,253 numbers. No source code, no game, no documentation of what
the network was trained on — just weights. By looking at *where* the numbers were large and where they
were silent, the game was reconstructed from first principles:

![Weight X-ray](assets/xray.png)

- **Agent channel** — near-zero everywhere except **row 8**: the network only cares where the paddle sits.
- **Moving-object channel** — weight mass blankets the ball's flight zone, and together with the
  previous-frame channel it encodes *velocity*, not just position.
- **Reward channel** — concentrated on **rows 1–3**: a stationary brick wall.

Conclusion: **Breakout**. Nobody told us. The knowledge was written into the numbers, and we read it
like an X-ray.

### Act 2 — The numbers still work, anywhere

The weights were exported as a PyTorch model, pushed to
[Hugging Face](https://huggingface.co/HarleyCooper/grid-policy-mlp), downloaded fresh, and verified
end-to-end. Then they were ported to ~40 lines of pure JavaScript — `tanh` matrices and a softmax —
so the *same decimals* now play in any browser tab:

![Live visualizer](assets/gameplay.png)

The visualizer renders the game the way the network sees it: five binary channels (agent, ball, bricks,
walls, previous frame), live softmax probabilities for all five actions, and a CRT arcade skin in honor
of the 1976 original.

### Act 3 — Surgery instead of schooling

If a policy makes a mistake, you don't have to retrain it from scratch. Ask a counterfactual question —
*which change to which numbers would have prevented that failure while preserving everything else it does
well?* — solve for the change, and verify by replaying the game:

![Weight surgery](assets/surgery.png)

Three deliberate faults injected into a hand-written Breakout policy — reversed controls, a left bias, a
severed "move right" connection — and each one repaired with **exact scalar edits, zero gradient steps**,
scored on 32 held-out seeds by the deterministic physics itself. That repair loop is the published
benchmark [weight-space-breakout](https://app.primeintellect.ai/dashboard/environments/harleycooper/weight-space-breakout),
where an LLM receives the broken weights and must emit the fix as JSON.

> **Honest note:** the weights playing in the demo are the original trained ones, unmodified. The repair
> half of the story lives in the benchmark above — the demo is the *before* picture that makes the
> *after* meaningful.

## Why it matters

Normally, changing what an AI does means retraining: expensive, slow, and you can never be sure what else
changed. Weight-space editing promises something different — behavior as **inspectable, addressable
mathematics**. Find the numbers responsible for a behavior, change exactly those, verify nothing else moved.

And if repair-by-reasoning works on a game, it works on any process with checkable steps: a railroad
engine manual's worth of structured knowledge becomes hyperspecific behavior for the cost of electricity.
Today this works on a 17-thousand-parameter toy. Whether it scales to billion-parameter models is the
open frontier.

## The cast

| Piece | What it is | Where |
|---|---|---|
| `GridPolicyMLP` | The original PPO-trained policy (500→32→32→5, tanh, 17,253 params) | [Hugging Face](https://huggingface.co/HarleyCooper/grid-policy-mlp) |
| This repo | Zero-build static visualizer; pure-JS inference | [live demo](https://harleycoops.github.io/breakout-policy-visualizer/) |
| `kimi-grid-breakout` | Closed-loop LLM Breakout environment (verifiers v1) | [Prime Intellect](https://app.primeintellect.ai/dashboard/environments/harleycooper/kimi-grid-breakout) |
| `weight-space-breakout` | Weight-repair benchmark + the authoritative physics | [Prime Intellect](https://app.primeintellect.ai/dashboard/environments/harleycooper/weight-space-breakout) |

## Run it locally

```bash
git clone https://github.com/HarleyCoops/breakout-policy-visualizer
cd breakout-policy-visualizer
npm run dev    # → http://localhost:7100/
```

Any static file server works — there is no build step. Controls: **Space** play/pause · **S** step ·
**R** reset · speed slider · sampled vs. deterministic policy · optional arcade sound.

## How the inference works

```js
h1 = tanh(x @ W1 + b1)   // 500 → 32
h2 = tanh(h1 @ W2 + b2)  // 32 → 32
logits = h2 @ W3 + b3    // 32 → 5  (noop / up / down / left / right)
```

`weights.json` is the whole model — 17,253 floats exported from the checkpoint. The forward pass above,
plus a softmax sampler, plus the grid physics, is the entire runtime.

---

*Physics engine adapted from [weight-space-breakout](https://app.primeintellect.ai/dashboard/environments/harleycooper/weight-space-breakout) (MIT). Built with [Kimi](https://kimi.com).*
