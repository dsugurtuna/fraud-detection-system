# Why it's built this way

## The problem

A fraud team can review only a fixed number of transactions each month, and frauds differ enormously in value. A model that ranks by probability alone can fill the queue with small frauds and miss the expensive ones.

## Design choices

**Why rank by expected value (probability x amount)?** Because it is the quantity the review team is trying to maximise: money that would otherwise be lost. It also makes the trade-off explicit, since a moderate chance of a large fraud can outrank a near-certain small one.

**Why weight fraud rows by log amount during training?** Because it tells the model that missing a large fraud matters more than missing a small one. The log keeps a handful of very large transactions from dominating training.

**Why calibrate afterwards?** Because the weights (and class balancing) change the model's probabilities. Expected value needs probabilities that match observed fraud rates, so isotonic regression maps scores back on a period the model did not train on.

**Why keep a direct expected-value regressor as well?** Because it is a genuinely different approach, and a cheap one to try. Validation, not preference, decides which to keep.

**Why split by time, not at random?** Because in operation the model always scores the future. A random split lets it learn patterns from later months, which inflates every metric.

**Why out-of-fold risk encodings for the training rows?** Because a merchant's fraud rate computed from the very rows it describes includes each row's own label. The model then learns to trust the encoding far more than it should. Computing each fold's encodings from the other folds removes that leak; validation and test rows already used training-period statistics only.

**Why hide late fraud reports from earlier periods?** Because fraud is usually reported days or weeks after the transaction. Training on labels that would not yet have existed makes the model look better than it would be in practice.

**Why compare with random review at the same capacity?** Because "how much more do we catch than we would anyway?" is the question a fraud manager asks, and the answer depends on how many reviews there are.

## Questions worth asking

**"Isn't validation doing too many jobs?"**
Yes, and the code says so. Validation drives early stopping, calibration and model selection, so its numbers are optimistic. The test period is untouched until the end and is the only fair estimate. With more data I would split validation into a calibration period and a selection period, or use rolling-origin evaluation over several months.

**"Why should I believe the synthetic results?"**
You should not read them as performance claims. The synthetic data has a planted signal so that tests and the demo exercise every step and can check the pipeline beats random review. It proves the plumbing works, not that the model would work on real transactions. Real performance needs real data, a later test period and monitoring after deployment.

**"What happens when fraud patterns change?"**
The model will degrade, and nothing here detects it yet. Month-by-month capture on the test period is the first signal; feature and score drift checks are the next step on the roadmap. Retraining on a rolling window, with the same time-based discipline, is the usual response.

## What's next

- Separate calibration and selection periods, or rolling-origin evaluation.
- Precision and recall of the review queue next to value captured.
- A per-review cost, so capacity becomes a decision rather than an input.
- Drift checks on features and scores by month.
