# Practical shortlist for new Harmony pseudo-targets

Assessment dated 2026-09-30, on `harmony-feature-evaluation` at base
`46fecee79a778de2f3a59a9e27e73e75fe472443`. This filters the
[52-entry musical candidate catalog](harmony-feature-candidate-catalog.md).
Candidate IDs below refer to that catalog.

**Status: proposed extraction work, not implemented or benchmarked.** Tool
capabilities were checked against primary documentation and relevant repository
code. No packages were installed, audio downloaded, new targets extracted, or
models trained for this assessment. Existing v3 still uses 12 descriptors.

## Practical conclusion

Start with descriptors that preserve pitch relationships and temporal structure
without requiring a correct chord name for every moment. A reasonable first
audio pilot is an interval co-activation profile, sustained tonal-change timing,
and recurrence of ordered tonal patterns. These provide new information beyond
the existing 45 summaries and can share one CPU chroma extraction pass.

Chord-quality, progression, and local-key labels are a second stage after
teacher validation. Bass, inversions, voice leading, and richer chord types are
research options with substantially harder measurement problems.

“Can be calculated automatically” is weaker than “is a reliable musical
pseudo-label.” A deterministic chroma descriptor needs numerical and musical
behavior checks. A named chord/key estimate also needs reference evaluation.
Neither teacher agreement nor a high student R² proves musical correctness.

## What is available here

| Resource | Observed state | Consequence |
|---|---|---|
| Existing 45-column table, splits, raw metadata | Present in the repository | Table-derived comparisons can be prepared without locating MP3s. |
| Song audio | No `.wav`, `.mp3`, `.flac`, `.ogg`, or `.m4a` found under `data/` or `harmony_branch/data/` | A waveform pilot needs the team's existing audio mounted in its extraction runtime. This check does not claim that audio is absent from every machine or external folder. |
| Source extraction notebook | Present; references external Vast.ai/Drive audio and pins Librosa 0.11.0 | Preserve that recipe for reproducing existing targets. |
| Temporal extractor code | [features.py](../src/harmony_branch/features.py) contains CQT and harmonic-HPCP candidates | Reusable building blocks, not a generator for every new feature in this document. |
| Chord validation pipeline | Source mapping, bounded Essentia generation, evaluation, and decision scripts exist | Reuse the pipeline, but a real accepted teacher report is not established by this review. |
| Default `python3` | Python 3.14.7 on macOS arm64; NumPy, Librosa, Essentia, mir_eval, PyTorch, Demucs, Basic Pitch, madmom, and music21 not import-discoverable | Do not assume commands run in this interpreter. Other environments were not exhaustively searched. |
| Branch dependencies | [requirements.txt](../requirements.txt) pins Librosa 1.0.0, Essentia 2.1b6.dev1389, mir_eval 0.8.2 | Dependency declarations are not proof of installation or compatibility. This differs from the original table's Librosa version. |

Use an isolated extraction environment whose Python/package/platform combination
passes installation and a small audio smoke check. Linux is a sensible first
candidate for Essentia's documented pip path, but no environment is certified by
this document. Freeze resolved versions before comparing extractors; do not
silently regenerate the old table using a different Librosa release.
[Essentia installation documentation](https://essentia.upf.edu/installing.html).

The team has already agreed that training audio is under four minutes. Preserve
the established track/span policy and match targets to that audio; this is not
a request to reopen the duration decision. Temporal analysis uses continuous
audio intervals, never invented adjacency between separately sampled windows.

## Tool assessment

| Tool or route | What it supplies | Practical decision and boundary |
|---|---|---|
| Librosa CQT/chroma, optional HPSS/CENS | Pitch-class timelines and representation choices | First-wave CPU building blocks. Additional interval, event, and recurrence summaries are custom code. Freeze preprocessing and assess percussive contamination; smoothing can erase useful changes. [Chroma example](https://librosa.org/doc/0.11.0/auto_examples/plot_chroma.html). |
| Librosa recurrence + custom sequence summaries | Similarity relationships between feature frames/sequences | First-wave candidate. A recurrence matrix is not already a scalar loop score. Use order-preserving windows, exclude trivial matches, and bound frame count. [API](https://librosa.org/doc/main/api/generated/librosa.segment.recurrence_matrix.html). |
| Librosa beat tracking | Estimated beat times | Optional second stage for changes per beat. Does not establish downbeats or meter; reuse reliable rhythm-owner outputs where available. [API](https://librosa.org/doc/0.11.0/generated/librosa.beat.beat_track.html). |
| Essentia HPCP and Key | Pitch profiles and major/minor key-template estimates | Candidate alternative representation and tonal-profile teacher. Check C-first versus A-first order. Template choice matters; library strengths are not calibrated probabilities. [Key](https://essentia.upf.edu/reference/std_Key.html). |
| Essentia ChordsDetection / ChordsDetectionBeats | Major/minor chord estimates; optionally one per interbeat segment | Available CPU baseline, explicitly experimental. No native claim of reliable seventh/suspended/extended-chord labels. The current repo generator already wraps the frame-based version. [Frame-based](https://essentia.upf.edu/reference/std_ChordsDetection.html), [beat-based](https://essentia.upf.edu/reference/std_ChordsDetectionBeats.html). |
| Chordino / NNLS Chroma | Vamp plugin for bass/treble chroma, dictionary-based chord transcription and smoothing | Useful optional comparator; not a drop-in Python function here. Verify host/plugin build and dictionary on the target platform. Published Mac binaries are described as Intel, so do not assume native arm64 readiness. [Author documentation](https://isophonics.net/nnls-chroma). |
| madmom chord models | Learned chord features and CRF decoding for major/minor plus no-chord | Optional teacher comparator if its environment and model assets work. The inspected label mapping has 25 classes; it does not solve richer chord vocabulary. [Author source](https://github.com/CPJKU/madmom/blob/main/madmom/features/chords.py). |
| Essentia Dissonance / HighResolutionFeatures | Spectral roughness and pitch-grid deviation descriptors | Technically extractable, but defer as Harmony targets because instrumentation/noise can dominate. [Dissonance](https://essentia.upf.edu/reference/std_Dissonance.html), [pitch-grid features](https://essentia.upf.edu/reference/std_HighResolutionFeatures.html). |
| Demucs + a pitch estimator | Estimated bass/vocal/accompaniment stems, followed by separate pitch analysis | Later research; pretrained inference adds compute and separation artifacts. The original Demucs repo is archived and points to a maintainer fork with limited maintenance. Record model/checkpoint identity. [Repository](https://github.com/facebookresearch/demucs). |
| Basic Pitch / Librosa pYIN | Polyphonic note estimates / fundamental-frequency estimates | Not an accepted whole-mix transcription route. Basic Pitch recommends one instrument at a time; use pYIN where a single fundamental is a defensible assumption. [Basic Pitch](https://github.com/spotify/basic-pitch), [pYIN](https://librosa.org/doc/0.11.0/generated/librosa.pyin.html). |
| music21 | Symbolic chord/key/Roman-numeral analysis from supplied notes/chords | Useful after transcription, not an audio chord teacher. Symbolic tooling does not remove upstream measurement errors. [Roman analysis API](https://music21.org/music21docs/moduleReference/moduleRoman.html). |
| mir_eval | Comparison of estimates with reference chord/key annotations | Evaluation tool, not a pseudo-label generator. Select metrics matching the claimed chord vocabulary. [Chord metrics](https://mir-eval.readthedocs.io/latest/api/chord.html), [key metrics](https://mir-eval.readthedocs.io/latest/api/key.html). |

No proprietary inference service is necessary for the first-wave proposal.
Tool existence does not establish an acceptable runtime or target accuracy on
our songs; both are pilot measurements.

## Filtered list: which candidates survive the feasibility review?

| Disposition | Catalog IDs | Reason |
|---|---|---|
| Cheap table-derived comparisons | F01, F02, F03, F04 | Derivable from existing mean chroma, with template/uncertainty choices. Useful interpretation changes, not new underlying measurements beyond the 45. |
| First-wave CPU descriptors to pilot | F05, F11, F21, F24, F27, F28, F37, F38, F39, F41 | New temporal/relationship information; no complete note transcription required. Definitions and real-audio robustness still need checking. |
| Alternative summaries, not automatic extra dimensions | F12, F13, F23, F26, F40 | Can reuse first-wave intermediates but partly duplicate other candidates. Compare rather than concatenate by default. |
| Conditional on chord/key/beat teacher quality | F06, F07, F08, F14, F17, F19, F22, F25, F29, F30, F31, F33, F34 | Musical value is plausible, but upstream errors propagate. F25 beat-phase evidence is narrower than a downbeat/meter claim; F33/F34 are estimated transitions, not established cadences/functions. |
| Defer from the initial target set | F09, F15, F16, F32, F35, F36, F42, F43, F44, F45, F46, F47, F50, F51, F52 | Richer vocabulary, context, transcription, or source separation makes reliable labels considerably harder. Keep in the catalog; do not label them impossible. |
| Diagnostic or cross-branch analysis first | F10, F18, F20, F48, F49 | Reliability or timbral/tuning effects dominate the current justification. These are not assumed useless, but need stronger reason to become musical targets. |

## First-wave shortlist with explicit proposed outputs

All cost estimates below are relative engineering judgments, not measured
benchmarks. They assume reuse of one cached chroma timeline per track.

| Priority | IDs | Proposed output block | Extraction and incremental cost | Evidence needed before acceptance |
|---|---|---|---|---|
| 1 | F11 | Six normalized interval co-activation values | Custom products of chroma bins; low extra CPU | Expected behavior for known intervals and transpositions; resistance to instrument/harmonic contamination. |
| 1 | F21, F24 | Tonal event rate and segment-duration dispersion | Smoothed tonal novelty, peak/persistence rules; low/moderate CPU | Stable chord plus moving melody must not cause excessive events; repeatability under gain and reasonable parameter changes. |
| 1 | F37, F38 | Repeated tonal-sequence coverage and recurrence prominence | Stacked/ordered chroma similarities; moderate CPU/memory, capped resolution | Distinguish repeating sequences from shuffled sequences with matching pitch counts; avoid static-tone and fixed-k graph-density artifacts. |
| 2 | F05 | Local template-fit mean and variation | Fixed key-profile scoring over windows; low extra CPU | Template ambiguity, near-uniform input, and mixed/percussive audio handling. These remain fit scores until key accuracy is established. |
| 2 | F27 | Median tonal jump across accepted events | Before/after segment-profile distance; low extra CPU | Vary jump magnitude while holding event count constant; measure residual redundancy with existing movement. |
| 2 | F28 | Long-/short-lag tonal-change contrast | Fixed-lag chroma or tonal distances; low extra CPU | Separate fast ornamentation from slow development; declare tempo dependence and mask a degenerate denominator. |
| 3 | F39, F41 | Longer-window contrast / long-range return score | Coarse windows and long-lag comparisons; moderate CPU | Verify that results reflect tonal organization rather than duration, segmentation choices, or energy changes. |

### Draft ten-value pilot bundle

Identifier: `candidate_beyond45_pilot10`. This makes the priority-1 proposal
concrete; it is not a frozen model interface or proof that ten dimensions are
optimal. Do not automatically append all ten to every old feature.

| Order | Proposed target names | Definition |
|---|---|---|
| 1–6 | `interval_coactivation_01` through `interval_coactivation_06` | Time-mean normalized within-frame co-activation at interval classes 1–6 semitones. |
| 7 | `tonal_event_rate_per_minute` | Accepted persistent tonal-change events divided by eligible continuous audio minutes. |
| 8 | `tonal_segment_duration_dispersion` | Interquartile range of eligible complete segment durations divided by their median. |
| 9 | `tonal_sequence_repeat_coverage` | Eligible sequence-window fraction having a non-overlapping, sufficiently similar ordered match elsewhere. |
| 10 | `tonal_recurrence_prominence` | Contrast of a supported recurrence-lag peak against its surrounding lag baseline, using a fixed normalized definition. |

For the interval profile, let `p_i(t)` be nonnegative L1-normalized chroma.
For each frame and interval class `k`, sum `p_i(t) * p_j(t)` over unordered
distinct pairs `i < j` with `min(|i-j|, 12-|i-j|) = k`. Normalize the six sums
by their total on eligible frames, then average over time. A frame with
effectively zero pair mass cannot define this normalized profile and must be
masked, not replaced by a uniform vector. This is co-activation of extracted
pitch classes, not an assertion that those notes were played simultaneously.
The six entries sum to one and are not six independent variables; F12/F13
would be alternative compressions, not additional information.

Event targets require frozen smoothing, novelty, minimum-persistence, and
threshold rules. Exclude invalid gaps and do not treat their edges as events.
A valid constant passage has event rate zero, while no usable audio is missing.
Duration dispersion requires enough complete segments and a nonzero median;
unknown is not zero. Missing intervals must not be bridged as long chords.

Recurrence uses short ordered sequences rather than isolated identical chroma
frames. Exclude self/near-neighbour and overlapping matches. Register the lag
range, similarity rule, threshold, minimum sequence duration, and handling of
static harmony. A graph constructed with exactly k neighbours per node has
largely predetermined edge density: **that density is not a repetition score**.
Use actual similarity and sustained sequence matching. A flat valid lag curve
has no prominent peak; inadequate coverage is a masked measurement.

Freeze the precise recurrence-prominence formula and all thresholds before
materializing a pilot table. These unresolved choices mean the bundle is
reviewable but not yet a reproducible production extractor. Long-lag targets
must also be predictable from the model's sampled audio; compare full-span and
model-visible-region targets before assuming supervision alignment.

After the pilot, compare whole blocks for incremental value rather than dropping
one coordinate of a structured six-bin profile based only on pairwise r.
The earlier [seven-value candidate](feature-selection-evaluation.md) remains a
separate low-cost comparator derived from the old table. It is not the endpoint
of this broader search.

## Second stage: what a validated teacher would unlock

| Prerequisite | Candidate block | Shape/semantics to freeze |
|---|---|---|
| Accepted major/minor chord timeline | F14, F17 | Duration-weighted quality composition and vocabulary entropy; retain accepted coverage and supported no-chord behavior. |
| Accepted root sequence | F29, F30, F31 | Directed relative-root transitions, conditional entropy, and compact repeated-pattern summaries; merge repeated states and exclude uncertain gaps. |
| Reliable beats plus tonal/chord events | F22, narrower F25 | Changes per beat and phase alignment. Missing beats are not a zero-change song. |
| Accepted local-key timeline | F06–F08 | Sustained change rate, dwell, and distance. Fit-score fluctuation alone does not establish modulation. |
| Accepted chord and key timelines | F33, F34 | Explicit V–I/IV–I/ii–V–I pattern counts, normalized by opportunities; phrase-level cadence claims remain outside scope. |

Do not add richer chord types by expanding a label list around a teacher that
cannot recognize them. A new dictionary or pretrained model requires separate
vocabulary-specific evaluation, not just confirmation that inference runs.

## Evaluation without new human annotation

1. **Deterministic and synthetic checks:** generate known notes, intervals,
   chord patterns, static accompaniment with changing melody, transpositions,
   and repeats/shuffles. Test expected behavior and masks. These checks expose
   errors but do not prove quality on mixed songs.
2. **Bounded real-song pilot:** preselect a small training-only sample across
   the six genres and available instrumentation, with several continuous
   regions per song. Include low-tonality/percussive examples. Preserve artist
   separation when practical and never tune thresholds on genre test scores.
3. **Extraction robustness:** record runtime, peak memory, failures, accepted
   coverage by genre, and sensitivity to gain, trimming, small timing changes,
   alternative chroma preprocessing, and transposition where relevant.
   Agreement between extractors is a diagnostic, not ground truth.
4. **Existing reference datasets for semantic teachers:** evaluate chords,
   keys, and segmentation against released annotations with matching audio.
   No team member has to annotate songs manually. Match metrics to vocabulary,
   assess boundary errors and no-chord behavior, and report conditional
   accuracy together with rejected/unknown coverage.
5. **Training-only feature review:** inspect distributions, duplicate/complement
   relations, nonlinear dependence, reliability, and overlap with other
   branches. Low correlation is not the goal by itself.
6. **Handoff to Dehan:** supply pseudo-targets, masks, definitions, provenance,
   and pilot evidence. Dehan evaluates student learnability and incremental
   validation genre macro AP using matched controls. No student training is
   part of this preparation task.

One concrete external option is the
[Schubert Winterreise Dataset](https://zenodo.org/records/5139893), which lists
song audio plus chord, local/global-key, and structure annotations. It is useful
for full musical recordings, but its voice/piano classical domain does not
establish performance across our six genres. An adapter is needed for the
released annotation formats; our existing benchmark runner expects its own
mapping and interval format.

[ChoCo](https://github.com/smashub/choco) provides a catalog of chord-annotation
partitions spanning, for example, rock/pop and jazz as well as classical music.
Annotations or audio identifiers do not guarantee downloadable matching audio.
Check each source's audio availability, alignment, and usage terms before
selecting a benchmark. An isolated-guitar-only evaluation would be inadequate
for the intended mixed-song use case.

## Resource and artifact plan

- Reuse one versioned waveform-to-chroma cache for several candidate summaries.
  Persist timestamps, valid regions, sample rate, frame/hop settings, tuning,
  preprocessing, and pitch order. The 45-column CSV cannot recreate a timeline.
- Run a CPU pilot first and measure cost before full-corpus extraction. Coarsen
  recurrence features in a declared way before pairwise comparison; dense
  similarity at the original frame rate grows quadratically with track length.
- Keep mixture-based chroma as a reference when assessing HPSS or smoothing.
  Neither separation nor a learned stem model automatically improves targets.
- Store one record per track with named raw features, per-feature masks, units,
  eligible durations/event counts, source hashes, extractor/model versions,
  parameters, failures, and runtime. Unknown and genuine zero remain distinct.
- Preserve benchmark separation from genre evaluation. A pretrained teacher
  may have seen a benchmark; record known training-data overlap rather than
  presenting such results as independent validation.
- Fit student-target normalization on training rows only. Changes to target
  order, dimension, masking, loss, or fusion projection require a new versioned
  contract before integration. No target family is required to have 12 values.

Existing scripts available for reuse are
[benchmark_harmony_extractors.py](../scripts/benchmark_harmony_extractors.py),
[prepare_chord_benchmark_source.py](../scripts/prepare_chord_benchmark_source.py),
[generate_essentia_chord_estimates.py](../scripts/generate_essentia_chord_estimates.py),
[evaluate_chord_teacher.py](../scripts/evaluate_chord_teacher.py), and
[decide_chord_teacher.py](../scripts/decide_chord_teacher.py).
Their existing resource caps remain in force when used. They do not implement
the proposed ten-value bundle, and their presence is not proof that a teacher
has passed the external benchmark.
