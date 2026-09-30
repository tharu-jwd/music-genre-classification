# Harmony feature candidates beyond the existing 45

Research catalog, 2026-09-30. These are hypotheses for the six-genre project
(classical, electronic, folk, hiphop, jazz, rock), not implemented targets or
demonstrated genre predictors. No new human annotation is planned.

Read this for the broad musical reasoning, then use the
[practical extraction shortlist](pseudo-label-feasibility-shortlist.md) to see
which candidates are realistic next steps. The
[earlier feature evaluation](feature-selection-evaluation.md) covers the
existing 45 descriptors and proposed seven-/nine-value comparisons. The
[current v3 contract](integration-handoff.md) is unchanged.

## Why broaden the search

The existing table summarizes pitch-class activity, tonal coordinates, and
adjacent-frame movement. It loses much of the information about simultaneous
intervals, repeated progressions, harmonic pacing, tonal organization, bass,
and voicing. Choosing only among those 45 columns limits the musical questions
the branch can answer.

Theoretical examples of complementary information:

- C–E–G and C–E-flat–G can have equal pitch concentration, but different chord
  quality. Note relationships matter in addition to the distribution's shape.
- Fast melody over a held chord can create chroma flux without a chord change.
  Harmonic pacing asks a different question from general pitch activity.
- A repeating progression and a continually developing sequence can have
  similar average pitches and local movement. Their ordering differs.
- A bass pedal under changing upper harmony differs from a sequence whose
  bass follows every chord root, even with similar whole-song chroma.
- A piece can have clear local keys while changing between them; its global
  average alone may look ambiguous.

These are musical distinctions worth testing, not rules such as “jazz must
have complex chords” or “electronic music must repeat.” Every target must show
incremental value alongside the instrument, rhythm, and timbre branches.

## Reading the catalog

There are **52 candidate entries**, some scalar descriptors and some vector
families. This is not a recommendation for a 52-dimensional model.

| Route | Meaning |
|---|---|
| Table | A new interpretation/derivation of existing columns; no new underlying audio information |
| CPU pilot | Plausible automated descriptor from a new chroma/audio timeline; custom summaries still need implementation and checks |
| Teacher gate | Requires reliable chord, key, or beat estimates before musical labels are defensible |
| Research | Strong assumptions, source separation, richer transcription, or substantial new engineering |
| Diagnostic | Useful for extraction quality or adjacent branches; not recommended as an initial Harmony target |

The judgments below are our theoretical assessments. Source links establish
tool capabilities, not evidence that these candidates improve this dataset.

## A. Tonal organization and pitch vocabulary

Fixed major/minor key templates can be scored from pitch profiles. Local
estimation requires temporal features. [Essentia Key](https://essentia.upf.edu/reference/std_Key.html)
provides key/scale estimates and match strengths; the strengths are not
calibrated probabilities. Template families have repertoire assumptions, so
do not choose a different template using a song's true genre label.

| ID | Candidate | Musical information and reason to consider it | Route and limitation |
|---|---|---|---|
| F01 | Global key-profile fit | Whether the arrangement of pitch activity resembles a coherent tonal centre; two distributions with the same entropy can fit a key differently. | Table. Global averaging can hide modulation; retain as an inexpensive comparison. |
| F02 | Major/minor profile preference | Distinguishes alternative scale organizations without tying them to one absolute key. | Table. A signed template-score difference, not a claim about mood or every chord's quality. |
| F03 | Tonic-relative pitch profile | Which scale degrees dominate, rather than whether the recording happens to use C or D. | Table. Estimated tonic can be wrong or ambiguous; 12 bins form a constrained distribution. |
| F04 | In-scale or out-of-scale pitch mass | Amount of pitch activity outside an estimated scale; could distinguish diatonic and chromatic vocabulary. | Table/global or teacher gate/local. These two fractions are complements: do not keep both as new information. Harmonic overtones also contribute. |
| F05 | Local key-fit stability | Typical local template fit and its variation distinguish consistently clear passages from alternating clear/ambiguous passages. | CPU pilot with soft scores. A key-template match statistic is more defensible initially than a hard modulation claim. |
| F06 | Sustained tonal-centre change rate | How often a piece shifts its local tonal centre, a longer-scale property than adjacent chroma movement. | Teacher gate. Require persistence and exclude ambiguous estimates; chord changes are not necessarily key changes. |
| F07 | Tonal-centre dwell distribution | Whether the piece stays in one centre for long spans or frequently moves between centres. | Teacher gate. Partly overlaps F06; censor uncertain intervals rather than merging across them. |
| F08 | Distance between successive tonal centres | Close moves and distant moves can occur equally often yet describe different tonal organization. | Teacher gate. Freeze a musical distance definition; depends on both key estimates being reliable. |
| F09 | Modal/pentatonic/blues template evidence | Can describe pitch organization outside a simple major/minor choice. | Research. Templates are implementable, but template fit is not proof of a mode or blues usage; confounding with relative keys is substantial. |
| F10 | Key ambiguity / estimator disagreement | Identifies where tonal claims are unstable or incompatible across estimators. | Diagnostic. Prefer reliability metadata; do not treat uncertainty itself as established musical complexity. |

## B. Simultaneous intervals and chord vocabulary

Pitch-class co-activation is an audio descriptor, not a transcription of every
played note. Harmonics and instrumentation affect it. Exact chord names need
a teacher and vocabulary checks. [Essentia ChordsDetection](https://essentia.upf.edu/reference/std_ChordsDetection.html)
matches major/minor triads and is explicitly documented as experimental.

| ID | Candidate | Musical information and reason to consider it | Route and limitation |
|---|---|---|---|
| F11 | Interval-class co-activation profile | How much pitch activity occurs together at separations of 1–6 semitones, independent of absolute key. Describes relationships lost by entropy. | CPU pilot. A six-bin chroma co-activation distribution, not six true note-interval counts. |
| F12 | Minor-/major-third balance | A compact contrast between two interval relationships that can differ despite equal pitch concentration. | CPU pilot, derived from F11. Alternative compression of that profile; no extra information if the full profile is retained. |
| F13 | Fifth/fourth versus tritone balance | Contrasts selected interval relationships and might summarize different harmonic vocabulary. | CPU pilot, derived from F11. Shares information with F11; distortion/harmonics can dominate. |
| F14 | Major/minor chord time proportions | Describes common chord qualities across a track rather than only a global key. | Teacher gate. Weight by accepted duration; distinguish unknown coverage from real no-chord audio. |
| F15 | Suspended, power, diminished, augmented chord proportions | Captures chord families omitted by a major/minor vocabulary, potentially useful across styles. | Research. Requires a verified richer teacher or carefully named template evidence; a triad-only teacher cannot produce these labels. |
| F16 | Seventh and extended-chord evidence | Adds chord colour that a triad-only view removes. | Research. Upper melody notes can be mistaken for extensions; no reliable richer teacher has been accepted here. |
| F17 | Duration-weighted chord vocabulary diversity | Whether harmonic time is concentrated in a small vocabulary or spread among more chords. | Teacher gate. Entropy is preferable to an unnormalized unique-chord count; detector flicker falsely increases diversity. |
| F18 | Chord-template residual / match margin | Measures how well the chosen vocabulary accounts for a frame. | Diagnostic initially. Low fit can mean noise, a missing chord type, or genuine ambiguity; not automatically complexity. |
| F19 | Chord-tone alignment of chroma | Fraction of pitch activity consistent with accepted local chord tones; relates accompaniment and non-chord activity. | Teacher gate. Reusing the same chroma to select and score a chord can make this a near-duplicate of fit; check F18 overlap. |
| F20 | Sensory roughness/dissonance | Captures interactions between spectral partials, different from a count of chord changes. | Diagnostic/cross-branch research. Strong dependence on timbre and register; not equivalent to tonal tension. |

## C. Harmonic pacing and timing

Time-domain summaries need a retained timeline and explicit units. First use
seconds; beat-relative targets additionally depend on beat quality. A change
in a smoothed chroma pattern must be named a tonal-change event until chord
recognition is validated.

| ID | Candidate | Musical information and reason to consider it | Route and limitation |
|---|---|---|---|
| F21 | Sustained tonal-change events per minute | Counts persistent changes instead of averaging all frame differences; separates event frequency from magnitude. | CPU pilot using novelty and persistence. Melody/transients may still create events; not automatically chord-change rate. |
| F22 | Harmonic changes per beat | Harmonic pacing relative to pulse; can differ for two songs with the same BPM. | Teacher gate for beats, plus event quality. Distinct from the rhythm branch's tempo; mask unreliable beats. |
| F23 | Median stable-segment duration | Measures sustained harmonic texture versus short-lived patterns. | CPU pilot as tonal segments; teacher gate as chord dwell. Often overlaps F21, so keep as an alternative unless useful. |
| F24 | Stable-segment duration variability | Distinguishes regular harmonic pacing from irregular holds at similar average rates. | CPU pilot. Use robust dispersion and sufficient events; edge-censored segments need explicit treatment. |
| F25 | Alignment of changes to beats/downbeats | Whether changes coincide with the pulse or occur between it; connects harmonic timing to meter. | Teacher gate for beat phase; research for downbeat/meter claims without a validated downbeat model. |
| F26 | Clustering/burstiness of harmonic changes | Distinguishes changes concentrated in short passages from evenly spaced changes. | CPU pilot. Some burstiness formulas are just transforms of duration CV; choose one, not redundant copies of F24. |
| F27 | Tonal jump size conditional on an event | Large and small harmonic shifts can occur at the same event rate. | CPU pilot using accepted novelty boundaries and before/after tonal profiles. Avoid reintroducing ordinary frame flux under a new name. |
| F28 | Short-versus-long-timescale change | Separates fast pitch activity from slower tonal development. | CPU pilot with several fixed time lags. Temporal scales and smoothing must be frozen; tempo can influence it. |

## D. Progression order and harmonic syntax

These require reliable symbolic sequences or explicitly limited proxies.
They are potentially informative because means and histograms ignore order.
Rule-based symbolic processing does not fix errors in audio recognition.

| ID | Candidate | Musical information and reason to consider it | Route and limitation |
|---|---|---|---|
| F29 | Directed chord-root interval transitions | Which relative root moves occur, independent of absolute key, rather than just how far tonal coordinates move. | Teacher gate. Merge repeated labels, preserve direction modulo 12, and do not bridge uncertain gaps. |
| F30 | Conditional chord-transition entropy | How predictable the next chord is given the current one; identical chord histograms can have different sequence structure. | Teacher gate. Sparse transitions need fixed smoothing and adequate coverage. |
| F31 | Repeated short progression dominance | How much of the accepted sequence is explained by repeated two-/three-/four-event patterns. | Teacher gate. Normalize transposition and count rules; an unrestricted n-gram vocabulary will be sparse on this dataset. |
| F32 | Relative harmonic-function profile | Use of tonic, predominant, and dominant-like functions rather than absolute chord names. | Research. Requires stable key and context; a Roman-numeral string alone is not a reliable functional analysis. |
| F33 | V–I and IV–I transition proportions | Tests specific relative progression tendencies. | Teacher gate for a narrowly named transition proxy; research for genuine cadences, which also require phrase context. |
| F34 | ii–V–I sequence proportion | A specific ordered relationship potentially complementary to chord-quality counts. | Teacher gate/research. Treat as an estimated pattern, not a genre rule; relative-key errors propagate. |
| F35 | Secondary-dominant / borrowed-chord usage | Describes departures from a basic diatonic vocabulary with tonal context. | Research. Needs accurate chords and local key, plus contextual interpretation; unavailable from basic chroma summaries. |
| F36 | Tension–resolution trajectory | Whether tonal distance builds and resolves across phrases. | Research. A chosen tonal-distance model is a proxy; it is not a direct measure of perceived tension or emotion. |

## E. Repetition and organization across time

These can begin with chroma sequence comparisons and need no chord names.
Sources provide building blocks: [Librosa recurrence](https://librosa.org/doc/main/api/generated/librosa.segment.recurrence_matrix.html)
and [AudioLabs novelty segmentation](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html).
Our proposed target summaries still require implementation and validation.

| ID | Candidate | Musical information and reason to consider it | Route and limitation |
|---|---|---|---|
| F37 | Repeated tonal-sequence coverage | Fraction of a song whose ordered tonal pattern reappears elsewhere; distinguishes looping from developing material. | CPU pilot with stacked chroma. Exclude trivial self/neighbour matches and distinguish sustained static harmony from sequence repetition. |
| F38 | Dominant recurrence lag and its prominence | Characteristic time between repeated patterns and how strongly one period dominates. | CPU pilot. A flat similarity curve has no meaningful period; physical seconds and beat-relative periods are separate targets. |
| F39 | Tonal contrast between longer sections | Changes in tonal vocabulary across large sections even when the whole-song histogram is similar. | CPU pilot using fixed windows initially. Do not label windows verse/chorus without structural evidence. |
| F40 | Tonal structural-boundary density | How often a longer-scale tonal pattern changes, instead of a short frame or chord event. | CPU pilot. Different from F21 only if time scale/definition differ; otherwise it is a duplicate. |
| F41 | Return to an earlier tonal region | Departure-and-return organization, beyond adjacent change or local repetition. | CPU pilot with long-lag similarity. Time ordering and a defined exclusion gap matter; duration affects opportunities for a return. |

## F. Bass, voicing, and interaction between parts

These are musically attractive but considerably harder in mixed recordings.
Bass chroma is not automatically an accurate bass-note transcription; source
separation is also imperfect.

| ID | Candidate | Musical information and reason to consider it | Route and limitation |
|---|---|---|---|
| F42 | Bass-to-chord-root agreement | Whether bass reinforces roots or moves independently. | Research/limited pilot using bass chroma plus a chord teacher; low-frequency drums and octaves confound it. |
| F43 | Chord inversion proportions | Bass placement distinguishes root-position and inverted versions of the same pitch-class chord. | Research. Needs reliable bass and chord labels together; octave-folded chroma alone is insufficient. |
| F44 | Pedal bass under changing upper harmony | A sustained/repeated bass centre with evolving upper parts captures a different relation than overall chroma movement. | Research/limited pilot with separate bass and upper chroma; require confirmed upper change rather than one static drone. |
| F45 | Bass step/leap and root-motion patterns | Describes the bass line's movement relative to harmonic events. | Research. A monophonic bass estimate may be possible after separation, but doubled/polyphonic bass and leakage remain. |
| F46 | Voice-leading smoothness | Small movements of individual voices across different chords can distinguish textures with identical chord vocabularies. | Research. Minimum matching of pitch-class sets is only a proxy; true voice leading requires voice identity and register. |
| F47 | Melody–chord agreement / non-chord-tone occupancy | Captures how a lead line interacts with accompaniment. | Research. Requires aligned melody and chord estimates, and assumptions about lead identity; cascading errors are likely. |

## G. Tuning, register, and texture at the branch boundary

These may add audio information but overlap with timbre/instrument identity.
Their existence is not a reason to assign them to Harmony automatically.

| ID | Candidate | Musical information and reason to consider it | Route and limitation |
|---|---|---|---|
| F48 | Tuning offset and temporal drift | Pitch placement relative to an equal-tempered grid and its change over time. | Diagnostic/research. Global tuning already exists as raw extractor metadata; drift requires new analysis. Recording speed and instruments can dominate. |
| F49 | Non-tempered pitch energy | Activity between the expected pitch grid locations could describe bends or tuning systems. | Diagnostic/research. Vibrato, noise, and instrument partials confound interpretation; needs higher resolution than 12-bin chroma. |
| F50 | Pitch register and voicing span | Octave placement and spacing are lost by chroma, despite possible differences in harmonic texture. | Research/cross-branch. Needs octave-aware pitch activation; overlaps instrumentation and recording arrangement. |
| F51 | Arpeggiated versus simultaneous chord texture | The same chord's notes played sequentially versus together can characterize accompaniment style. | Research. Requires reliable onset/pitch alignment and separation of accompaniment from melody. |
| F52 | Independent-voice / polyphonic density | Simultaneous independent lines can differ despite comparable pitch-class concentration. | Research. Counting chroma peaks or entropy does not estimate the number of voices; mixed-audio transcription is the hard dependency. |

## What not to mistake for new information

- A different chroma extractor, HPSS, CENS smoothing, or a new normalization is
  a representation/quality choice, not automatically a new musical concept.
- F12/F13 compress F11; F04's in-/out-of-scale fractions are complements;
  F21/F23 may be almost reciprocal for uniform durations; F26 may duplicate
  F24. Keep alternatives visible without stacking them blindly.
- Absolute key/root labels can encode recording or repertoire biases. Prefer
  musically justified relative representations where possible, while retaining
  masks for uncertain tonic estimates.
- “No chord,” silence, invalid extraction, and low teacher confidence are
  distinct conditions. Neither invalid audio nor uncertainty proves atonality.
- A pretrained genre tagger's output or unrestricted audio embedding is not a
  named Harmony concept. It would change the interpretability claim.

The next practical selection, tool constraints, and pilot gates are specified
in the [feasibility shortlist](pseudo-label-feasibility-shortlist.md).
