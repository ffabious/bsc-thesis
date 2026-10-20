# Thesis idea: faster recommendation models

Make large recommendation models fast enough to use fresh user activity.

## What I found

- ARGUS does expensive work offline and stores user representations for quick online scoring. [Paper](https://arxiv.org/html/2507.15994v1)
- Pinterest selects relevant past actions before running a small transformer. [Paper](https://arxiv.org/html/2506.02267v1)
- Meta reuses history calculations when scoring multiple candidates. [Paper](https://arxiv.org/html/2402.17152v3)
- LinkedIn updates user representations as activity arrives, then uses recent history plus a long-term summary for online scoring. [Paper](https://arxiv.org/html/2607.28940v1)

## What we could investigate

Can we update a large recommendation model cheaply by keeping recent user actions, a small selection of important older actions, and saved calculations that we can reuse?

Compare this with processing the full history and with periodically refreshed offline representations. Measure recommendation quality, response time, and memory use.

## Where sliding-window attention helps

It limits attention to recent events, making it a useful starting baseline. The paper finds good results without retraining on language models, but that does not yet prove it works equally well for ARGUS. [Paper](https://arxiv.org/html/2608.28444v1)

## Possible contribution

A better rule for choosing what history to retain or when to refresh it. Reusing calculations is already established; we need a specific improvement to make this a research contribution.
