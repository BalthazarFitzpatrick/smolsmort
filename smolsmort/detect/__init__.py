"""the vision backend: finding objects of a fixed known size in frames.

ONE BACKEND, NOT THE PRODUCT. The loop smolsmort is - examples, human judgements, a named dataset,
named weights, predictions that re-enter judging - is model-agnostic. This package is the first
implementation of the model half of it, and a tabular one (xgboost, for forecasting) is meant to sit
beside it rather than replace it. Anything written here that assumes an IMAGE belongs here;
anything that would be equally true of a spreadsheet row belongs in the loop instead.

    box.py      a rectangle, and the two tests for "is this the same object again"
    model.py    the heatmap cnn, its targets and its loss
    dataset.py  training examples assembled from review decisions
    train.py    the training loop, and `sweep` - running weights over whole frames
    scoring.py  precision and recall against hand-drawn truth
    track.py    following ONE object across a recording, given per-frame detections

NOTHING HERE KNOWS WHAT IT IS LOOKING AT. It works on boxes, tiles, classes and weights; what the
objects mean, which classes exist, and how to guess one are the consumer's business. The boundary is
worth defending: if a module here would stop making sense detecting something other than whatever
you happen to be detecting today, it belongs on the consumer's side of it. It was extracted from a
game-screen object detector, and the measured numbers that survive as DEFAULTS say so
explicitly wherever they appear - a default with a stated provenance is a parameter, one without is
a magic number.

ONE ASSUMPTION IS NOT GENERIC, and is called out where it lives: `model.py` exploits the object
being a FIXED KNOWN SIZE, which is why it predicts a centre heatmap and regresses no box at all.
That is a real constraint on what this backend can detect, so it is stated rather than hidden. A
consumer whose objects vary in size needs a different backend behind the same loop.
"""
