"""the size-aware vision backend: objects of varying size, in frames of varying resolution.

A SIBLING OF detect/, NOT A REPLACEMENT. detect finds objects of one fixed known size and regresses
nothing, which is the right tool when that holds. This backend drops the assumption and predicts each
object's own box, for a consumer whose objects vary in size - by distance, by class, or both.

    model.py      the centre-and-size cnn, its targets, its loss, the box decode
    train.py      the training loop, and `sweep` - running weights over whole frames
    synthetic.py  frames with known boxes, for proving the mechanics without real data
    backend.py    the ModelBackend adapter the loop drives

MEASURED LIMIT, stated rather than hidden: on synthetic frames it found 97% of objects standing
apart but only about half of those overlapping another. It is a starter to replace, behind the
same seam, when a consumer's objects crowd.
"""
