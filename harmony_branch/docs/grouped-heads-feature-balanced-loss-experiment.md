# Harmony V3 grouped-heads and feature-balanced-loss experiment

This experiment retains the public Harmony V2/V4 contract:

```text
encoded_sequence [B,T,128] -> harmony descriptors [B,45] -> fusion
```

It changes two internal training choices only.

1. The existing 18 exact outputs remain deterministic: 12 chroma means and 6 Tonnetz means.
2. The remaining 27 learned outputs are emitted by three independent heads:
   `chroma_std` (12), `tonnetz_std` (6), and `tonal_dynamics` (9).
3. Harmony loss is a fixed blend: 75% ordinary masked Smooth L1 across all
   45 descriptors plus 25% group-balanced Smooth L1. The latter is averaged
   within, then equally across, five groups: chroma mean, chroma std, Tonnetz
   mean, Tonnetz std, and tonal dynamics.

The feature names, order, 45-dimensional output, fusion ownership, target transforms,
and shared-CNN interface are unchanged. The 0.25 group-balanced contribution is
recorded in each checkpoint; it can be set explicitly with
``--harmony-group-balanced-weight``.

For the 30-epoch optimization run, the training entrypoint also supports a
Harmony-only optimizer group. For example, ``--lr 3e-4 --harmony-lr 5e-4``
keeps the shared encoder, the other branches, and fusion at ``3e-4`` while
training the Harmony trunk and descriptor heads at ``5e-4``. This changes no
public branch interface.
