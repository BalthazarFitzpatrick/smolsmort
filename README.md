# smolsmort

A human-in-the-loop sorting loop: find candidates, judge them, train on the judgements, predict, and
feed the predictions back in to be judged again.

The loop is the product. What you are detecting, how a person looks at one, and which model you fit
are all plugged in.

```
  data  ──FIND──▶  candidates ──cut──▶  tiles ──JUDGE──▶  classes
                                                            │
                                                         promote
                                                            │
                                                            ▼
   predictions  ◀──predict──  weights  ◀──TRAIN──  training set
         │
         └──cut──▶  tiles  ──▶  JUDGE   (the loop closes here)
```

**The loop closes at judging**, and that is the whole point. A wrong prediction becomes a labelled
hard negative rather than being thrown away, which is what makes the next model better instead of
merely retrained.

## Why this exists

It was extracted from a game-screen object detector, where the loop turned out to be the part
worth keeping and the objects the part that was not. Two things follow from that history and are
worth stating plainly:

- **Every measured default carries its provenance.** Numbers like a 40px "same object" tolerance
  were measured on a ~132x12 target bar. They are parameters with a stated origin, not facts —
  a default without its provenance is a magic number, and you should pass your own.
- **One lesson is baked into the design.** Judging a candidate is not the same as producing it. The
  pixels can be recut from a script; the human's verdict on them cannot be recovered from anything.
  The two are stored, versioned and ignored differently throughout.

## What is honest about it today

- The **vision backend** is real and used daily: a small heatmap CNN, ~50k parameters, that finds
  objects of a fixed known size.
- The **tabular backend** (xgboost) is planned, not written. The seams exist for it; the
  implementation does not.
- The **first honest evaluation of a trained model in the parent project returned 24% precision on a
  genuine holdout**, with the highest-scoring detections being the wrong ones. That is not a
  criticism of the loop — it is what the loop is for finding out, and the number was invisible until
  a hand-drawn holdout existed. Read a separation score as "did training converge", never as "is it
  right".

## Install

```bash
uv add smolsmort                 # the loop
uv add "smolsmort[vision]"       # ...and the CNN backend, which pulls torch
```

Torch is optional on purpose: the judging half does not need it, and a consumer doing tabular work
should not have to install it to use the rest.

## Status

Early. The package is being extracted in stages from its first consumer, and the API will move until
that is finished.

## Licence

MIT - see [LICENSE](LICENSE).
