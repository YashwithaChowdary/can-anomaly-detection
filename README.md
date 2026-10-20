# CAN Bus Anomaly Detection

Detecting faults and attacks on a vehicle CAN bus from traffic alone, using a
rule baseline and an unsupervised machine learning model, with both trained on
healthy traffic only.

The question I wanted to answer was not "can a model find anomalies", but
**which faults does each method actually catch, and which does it miss?**
The short answer is that the rules and the model fail in different places, and
both are nearly blind to the two faults that look like normal traffic.

## Results

120 seconds of traffic per run, 0.2 s windows, threshold set at a 2% false
positive rate on held-out healthy traffic. Detection rate is the share of
windows inside the fault that were flagged.

| Fault | Rules | Isolation Forest | Both | False alarms |
|---|---|---|---|---|
| dos (bus flood) | 100% | 100% | 100% | 2.7% |
| fuzzing (random ids) | 100% | 62% | 100% | 2.7% |
| suspension (node goes silent) | 1% | 72% | 72% | 2.7% |
| replay (stale frames re-sent) | 0% | 16% | 16% | 2.7% |
| drift (signal slides out of range) | 0% | 1% | 1% | 2.7% |

Reproduce with `py run.py`. Numbers move by a few points with the seed.

### What this says

- **The rules are not a weak baseline.** On a flood and on random ids they are
  perfect, instant and cost nothing. An id that has never been seen before is
  a decisive signal, and no model is needed to notice it. A machine learning
  model that cannot beat this is not worth putting on an embedded board.
- **The model earns its place on the silent node.** When an ECU stops talking,
  nothing unknown appears and nothing floods: the traffic just gets slightly
  thinner. The rules catch 1% of it. The forest catches 72%, because it has
  learned what a normal mixture of messages looks like.
- **Both nearly fail on replay and drift, and that is the real finding.** A
  replayed frame has a valid id, a plausible payload and roughly the right
  timing. A drifting signal changes nothing at all about the traffic pattern,
  only the number inside the payload. Window-level statistics cannot see
  either. Catching these needs per-signal models of what each byte normally
  does, which is a different and harder piece of work.
- **The two methods are complementary**, so the sensible deployment is both:
  the union row matches the better of the two on every fault.

The forest also reports which feature was furthest from normal in each flagged
window. It is not a true attribution, but it points an engineer somewhere
instead of handing them a bare alarm: `frames_per_s` for the flood and the
silent node, `unique_ids` for fuzzing.

## How the threshold is chosen

On a vehicle, faults are rare and healthy traffic is plentiful, so both
detectors are fitted on healthy traffic only and never see a labelled fault
during training. The decision threshold is taken from a **separate** capture of
healthy traffic, at a chosen false positive rate. A threshold tuned on data
containing the faults would produce a number the deployed system could never
reproduce.

The measured false alarm rate on faulty runs is 2.7% against a 2% target. The
gap is expected: the faulty runs contain healthy windows adjacent to the fault
that are partly contaminated by it.

## Files

| File | What it does |
|---|---|
| `can_bus.py` | Simulates a 12-message periodic bus and injects five faults |
| `features.py` | Turns frames into one row of ten numbers per 0.2 s window |
| `detect.py` | `RuleBaseline`, `ForestDetector`, and the evaluation |
| `run.py` | Trains on healthy traffic and prints the table above |
| `test_detect.py` | 24 unit tests |

## Run

    py -m pip install pandas scikit-learn numpy
    py run.py
    py run.py --target-fpr 0.05 --seconds 240

## Test

    py -m unittest test_detect.py

24 tests cover the simulator (timing, that each fault only touches its own
window, that drift changes payloads but not timing), the features (empty
input, fixed column order, that a flood raises the frame rate), and the
detectors (that the threshold matches the requested false positive rate, that
the forest beats the rules on a silent node, and that every fault runs end to
end).

## Limits

- **The traffic is simulated, not recorded from a vehicle.** The message
  periods, jitter and payload behaviour are modelled on how a CAN bus behaves,
  but real buses have bursts, bus-off events, diagnostic sessions and
  mode changes that this does not reproduce. Every number here should be read
  as a comparison between methods, not as a performance claim for real traffic.
- No Ethernet traffic yet. The feature extraction would carry over, but the
  decoding and the topology would not.
- No DBC decoding: payloads are treated as raw bytes rather than as named
  signals. That is precisely why drift is missed, and it is the first thing I
  would change.
- The forest is classical machine learning, not a time-series model or an
  autoencoder. That was a deliberate choice for a small memory budget, but it
  means the model sees each window independently and has no memory of the
  previous one.
