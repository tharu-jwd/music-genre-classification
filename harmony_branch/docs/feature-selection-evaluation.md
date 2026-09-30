# Harmony feature evaluation and candidate sets

Recorded 2026-09-30 against repository base
`46fecee79a778de2f3a59a9e27e73e75fe472443`.

**Status: findings and research proposals.** Harmony v3 predicted its
selected 12 descriptors. The all-45 candidate below is now implemented as an
untrained v4 option; the other proposed sets have not been implemented or
trained. No new target table was extracted. See the [v3 handoff](integration-handoff.md),
[45-feature contract](feature-contract.md), and
[v3 snapshot](architecture-versions/v3/README.md) for implemented behavior.

The broader [52-entry musical catalog](harmony-feature-candidate-catalog.md)
and [practical pseudo-extraction shortlist](pseudo-label-feasibility-shortlist.md)
extend this review beyond the existing 45 columns. They include a proposed
ten-value audio pilot and conditional chord/key/bass research. The seven- and
nine-value sets below are inexpensive comparisons, not a restriction on the
feature search or the recommended final target vocabulary.

## Purpose and conclusion

The project aims to predict genre through inspectable musical concepts. Harmony
should contribute useful information about pitch relationships and their changes,
beyond what instruments, rhythm, and timbre already contribute.

The existing 12 descriptors are a defensible reference baseline, but their exact
selection is not established as optimal. Several describe nearly the same
variation in the training data. Some omitted columns could supply complementary
information; therefore, reducing the set solely by dropping correlated or poorly
predicted targets is insufficient.

“Complementary” means answering a different musical question. It does not imply
statistical independence or mathematical orthogonality. High Pearson correlation
does not prove that a feature has no conditional or nonlinear value. Conversely,
low correlation does not prove that a feature is reliable or useful for genre.
Weak prediction scores may reflect label noise or model limitations as well as
target choice. Easy target prediction alone does not establish genre usefulness.

## Observed evidence

These diagnostics used the 5,127 training rows joined by normalized track ID from
[harmony_df.csv](../../data/harmony_df.csv) and
[track_split_assignments.csv](../../data/track_split_assignments.csv).
No genre labels, validation rows, or test rows were used for the correlation
or algebraic reconstruction checks.

| Target pair | Training Pearson r | Interpretation |
|---|---:|---|
| Concentration mean / entropy mean | -0.9741 | Strong overlap in pitch concentration |
| Concentration std / entropy std | 0.9641 | Strong overlap in concentration variation |
| Chroma flux mean / Tonnetz movement mean | 0.9879 | Strong overlap in local change magnitude |
| Chroma flux std / Tonnetz movement std | 0.9863 | Strong overlap in variation of local change |
| Tonnetz movement mean / movement std | 0.8250 | Even these two retained candidates overlap; they are not independent |

All six Tonnetz means were reconstructed from the 12 chroma means using
Librosa's fixed projection matrix. The largest absolute error across all six
coordinates and all training rows was approximately `1.54e-9`, consistent with
floating-point rounding. Under this extractor's common valid-frame mask and L1
normalization, `mean(Tonnetz) = projection × mean(chroma)`. Including both sets
does not add new measured information, although a transformed representation can
still change how easily a model learns it. This follows from the
[Librosa 0.11.0 implementation](https://librosa.org/doc/0.11.0/_modules/librosa/feature/spectral.html#tonnetz).

The training median `valid_tonal_ratio` is 0.9872; about 44.0% of tracks are at
least 0.99. Its definition is energy/chroma availability, not confirmed tonality.
The concern is its musical meaning, rather than a claim that it is constant.

Separately, the already supplied [test report](evidence/harmony_supervised_joint_run_results.json)
has movement-mean R² 0.8107, flux-mean R² 0.7962, and concentration-mean R²
0.7619. The three selected Tonnetz means have R² 0.0718, 0.0436, and -0.0042;
valid tonal ratio has R² 0.1384. Results are mixed, not uniformly poor.
These are existing test observations, not criteria for selecting new variants.
The reported whole-model genre macro AP 0.7356 does not establish Harmony's
contribution without a matched comparison.

### Diagnostic provenance

Pearson coefficients were calculated on raw finite feature values using the
ordinary centered-product formula. No feature standardizer was fitted for these
checks. Tonnetz reconstruction used the six cosine/sine rows and radii in the
linked Librosa implementation, applied to each track's 12 chroma means.

| Artifact | SHA-256 |
|---|---|
| `data/harmony_df.csv` | `308c76478eab584a854a9af6076d7b92ecbe71d6ea87b03fef7e80efdd935fee` |
| `data/track_split_assignments.csv` | `269b775cc4b48ff1c1c88f13a4da43e3cf4fdce3f5fadd90c5dd7c5d05961e10` |
| Supplied joint-run JSON | `f06df30563a5488daf18c4b194037e52c7333737058f8ed73472f15d70206a77` |

## Assessment of all 45 existing features

| Group | Count | Distinct information and theoretical judgment |
|---|---:|---|
| Chroma means | 12 | Which pitch classes dominate. They sum to approximately one, so there are at most 11 independent coordinates. Their arrangement can distinguish patterns having identical entropy. Absolute key is a possible shortcut; key-relative interpretation is a candidate. |
| Chroma standard deviations | 12 | Which pitch classes stay stable or vary across the track. Equal means need not imply equal variability. They lose order and depend on absolute pitch labels; consider key-relative values or an aggregate spread measure. |
| Tonnetz means | 6 | A projection of the chroma means, not additional measured information if all chroma means are present. Selecting the first three gives one complete coordinate pair and half another, with no established advantage. |
| Tonnetz standard deviations | 6 | Overall spread in tonal space. This differs from adjacent-frame movement. Individual chroma variances alone do not generally determine these values: the projection also depends on how pitch classes vary together. Pairwise spread summaries are promising candidates. |
| Concentration mean/std | 2 | Strength and variability of the largest chroma bin. Does not measure chord complexity or count simultaneous notes. Strong overlap with entropy in this dataset; retain as an alternative representation to compare. |
| Entropy mean/std | 2 | Spread and variability of the normalized pitch distribution. Does not identify interval relationships, chord quality, or major/minor mode. A reasonable representative of the concentration family. |
| Chroma flux mean/std | 2 | Magnitude and variability of adjacent-frame chroma changes. Melody, transients, and extraction noise can contribute; it is not a chord-change count. Strong overlap with Tonnetz movement here. |
| Tonnetz movement mean/std | 2 | Adjacent-frame change measured in tonal geometry. A reasonable movement-family representative, but not an established better genre predictor than flux. |
| Valid tonal ratio | 1 | Energy/chroma-availability coverage. Retain as diagnostic metadata; excluding it from musical targets is a hypothesis to evaluate, not a finding that it has no genre association. |
| **Total** | **45** | More columns do not automatically mean more distinct or useful information. |

## Concrete candidate A: seven descriptors from the existing table

Working identifier: `candidate_summary7`. This is a proposed target set, not a
registered production contract. The proposed order is below.

| # | Candidate target | Derivation | Musical question |
|---|---|---|---|
| 1 | `chroma_entropy_mean` | Existing column | How spread out is pitch activity at a typical moment? |
| 2 | `chroma_entropy_std` | Existing column | How much does that spread vary? |
| 3 | `tonnetz_movement_mean` | Existing column | How much does tonal position change between adjacent frames? |
| 4 | `tonnetz_movement_std` | Existing column | How variable are those local changes? |
| 5 | `tonnetz_fifths_spread` | `sqrt(tonnetz_01_std² + tonnetz_02_std²)` | How broadly does the track vary in the first tonal coordinate plane? |
| 6 | `tonnetz_minor_thirds_spread` | `sqrt(tonnetz_03_std² + tonnetz_04_std²)` | How broadly does it vary in the second tonal coordinate plane? |
| 7 | `tonnetz_major_thirds_spread` | `sqrt(tonnetz_05_std² + tonnetz_06_std²)` | How broadly does it vary in the third tonal coordinate plane? |

The first four provide a compact reference description of concentration and
local movement. The three spreads add a candidate description of overall
variation. Two sequences can visit the same tonal positions in different orders:
their spreads stay the same while their adjacent movement can change. Therefore
spread and movement measure different properties, though they may correlate in
real music.

Each spread is the square root of the sum of variances in a paired plane.
Under an ideal circular pitch-class transposition, those coordinates rotate
together and this quantity is invariant. Real waveform pitch shifting may also
change extraction artifacts, so practical robustness still needs checking.
These spreads are **not** estimates of the proportion of minor or major chords.
The coordinate naming follows [Librosa's Tonnetz definition](https://librosa.org/doc/0.11.0/generated/librosa.feature.tonnetz.html).

Compute these derivations from raw columns before fitting training-only target
standardization. The third plane uses a different radius in Librosa; raw units
and the fitted scaler must be recorded. No source audio or manual annotation is
needed to derive this set. No independence or genre improvement has been measured
for it. If the three spreads are redundant, compare an aggregate spread rather
than retaining three dimensions solely to reach a chosen width.

## Optional candidate B: global pitch-pattern information

Working identifier: `candidate_summary7_key2` (nine proposed values).
Append these two targets to candidate A only after the estimator is specified:

| Proposed target | Definition to freeze | Added question and limitation |
|---|---|---|
| `global_key_profile_fit` | Maximum correlation of the mean chroma vector with a fixed family of major/minor key templates across all roots | Does the pitch pattern resemble a coherent key profile? Unlike entropy, this depends on where activity occurs. It can be reduced by modulation or a poor template match. |
| `global_major_minor_profile_gap` | Best major-template correlation minus best minor-template correlation, each maximized over roots | Which template family fits better? This is a signed comparison, not a probability, a chord label, or a certainty that the song has one mode. |

These are derivable from the existing 12 chroma means. Before materialization,
freeze the exact templates, pitch order, correlation/normalization definition,
degeneracy handling, and uncertainty policy. Uniform or otherwise undefined
inputs need masks rather than invented labels. The two scores may themselves
overlap and must be checked.

An alternative is a key-relative 12-bin chroma profile, formed by rotating the
mean profile to an estimated tonic. It retains more detail but depends on tonic
accuracy; use an uncertainty-aware policy. Treat it as an alternative encoding
of pitch-pattern information, not an automatic addition to every key score and
Tonnetz mean. The pattern is derived information, not new evidence beyond the
original chroma means. [Essentia's Key documentation](https://essentia.upf.edu/reference/std_Key.html)
describes profile-based estimation and strength measures; those measures are not
automatically calibrated confidence probabilities.

## Candidate information requiring temporal audio features

These are independent research options, not one approved expanded vector. Exact
target definitions and extraction quality must be established before assigning
an output width. Whole-song means/stds cannot recover the discarded timeline.

| Information family | Potential targets and derivation | Why it may add value | Main risk |
|---|---|---|---|
| Chord quality / simultaneous pitch relationships | Time proportions of reliably estimated chord families, or a framewise pitch-interval co-occurrence profile summarized over time | Major and minor triads can have equal entropy/concentration but different musical relationships. Whole-song note averages cannot establish which notes sounded together. | Harmonics, melody, and teacher errors can resemble extra chord tones. Match the vocabulary to demonstrated teacher capability. |
| Harmonic pacing | Sustained harmonic changes per beat; median stable-segment duration; duration variability | Fast melody over a held chord differs from rapid changes in the underlying harmony. Per-beat timing can complement the rhythm branch's tempo. | Teacher flicker must not become false chord changes; beat errors and missing beats require explicit handling. Rate and duration can be redundant. |
| Harmonic recurrence / ordering | Repeated tonal-pattern coverage from chroma subsequence similarity; alternatively, conditional transition entropy from reliable chord sequences | A repeated loop and a developing progression can have similar averages, spreads, and chord counts but different ordering. Chroma recurrence does not require chord names. | Fix sequence scale, transposition policy, duration normalization, and coverage; avoid treating repeated percussion as repeated harmony. |
| Local tonal-centre stability | Sustained changes in estimated local key; agreement of local and global key profiles | A piece can be locally clear but move between tonal centres, making its global profile appear diffuse. | Ambiguous local estimates can masquerade as modulation. Require sustained evidence and retain uncertainty. |

For chord candidates, an algorithm limited to major/minor triads cannot establish
seventh, suspended, diminished, or extended-chord usage. An expanded teacher
requires validation on an existing annotated benchmark; no manual annotations
are required from the team. [Chord-template explanation](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S2_ChordRec_Templates.html).

[Essentia ChordsDescriptors](https://essentia.upf.edu/reference/streaming_ChordsDescriptors.html)
offers histogram and change summaries, but its change rate is not automatically
changes per beat or per second. Define the intended units explicitly.

All current descriptors are extracted from mixed audio without harmonic
separation. Harmonic separation and smoothing are potential extraction
comparisons, not assumed improvements. Excessive smoothing can erase the very
changes we want to measure. [Librosa's chroma enhancement example](https://librosa.org/doc/0.11.0/auto_examples/plot_chroma.html).

## Comparison plan and ownership

Tharupahan owns target reasoning and preparation; Dehan owns model training.
The team retains the agreed under-four-minute training-audio policy.

1. Preserve current v3 and its exact 12-feature order as the baseline.
2. Prepare train-only diagnostics for candidate A and the omitted feature
   groups: distributions, redundancy, invalid values, and expected behavior
   under transposition, reordering, and timing changes. Analysis of candidate
   usefulness should consider other branches as well as Harmony itself.
3. Compare the current model with a matched model trained without Harmony.
   Removing Harmony only at inference answers a different question.
4. If target comparisons are warranted, candidate A tests broader coverage
   with fewer duplicates. An eight-feature control (current 12 minus the three
   Tonnetz means and valid ratio) isolates those removals. A four-feature
   control (candidate A rows 1–4) isolates the addition of tonal spread.
   These controls are optional diagnostics, not a mandate for a full sweep.
5. Evaluate candidate B or a key-relative profile only if pitch-pattern
   information justifies the added estimation assumptions. Prioritize temporal
   recurrence and harmonic pacing for a later audio pilot; validate chord/key
   teachers before relying on their summaries.
6. Select with validation genre macro AP and branch diagnostics, keeping
   tracks, splits, encoder, optimization budget, and loss policy comparable.
   Report changes in head/projection size and parameter count; feature-count
   changes can also change auxiliary-loss weighting across musical families.
   Preserve the test split for the selected final comparison rather than
   repeatedly selecting targets against the existing test scores.
7. If a candidate is accepted, create a new versioned target/architecture
   record and agree with fusion on the exact target order, masks, scaler, and
   projection width. There is no requirement to retain 12 outputs.

The seven- and nine-value sets are concrete hypotheses for review. No claim is
made that they are mutually independent, optimal, or already improve genre
prediction. Reliability, musical coverage, and incremental genre value determine
whether a target should survive, rather than correlation or R² alone.
