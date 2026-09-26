# Intelligent Adaptive AI System — IAAIS

IAAIS is a CSC5350 course project exploring an intelligent adaptive AI system. The current concept may use wearable gyroscope and accelerometer data to recognize exercises, estimate repetitions, log workouts, and support user review.

## Planned capabilities

- Symbolic reasoning and knowledge representation
- Search and planning methods
- Probabilistic reasoning over uncertain sensor and classifier evidence
- Machine learning for sensor-based exercise recognition
- Natural language processing for workout interaction and review
- Generative AI features for adaptive assistance
- A Streamlit interface for demonstrations and user review

## Current implementation status

The repository began as an environment scaffold. The implemented course
components currently reach Chapter 6: the Chapter 2 Search Engine in
`src/iaais/search_engine/`, the Chapter 3 Knowledge Base in
`src/iaais/knowledge_base/`, the Chapter 4 Planner in
`src/iaais/planner/`, the Chapter 5 Uncertainty Module in
`src/iaais/uncertainty/`, and the Chapter 6 Classifier in
`src/iaais/classifier/`.

The Search Engine exposes a uniform problem interface:

- `initial_state` — the starting state;
- `goal_test(state)` — whether a state satisfies the goal; and
- `actions(state)` — the transitions available from that state.

The default strategy is bounded A*. Uniform-cost, breadth-first, depth-first,
and greedy best-first search are also available. A* is the intended IAAIS
default because the exercise-interpretation state space should be finite and
bounded, with a small branching factor and useful domain estimates for the
remaining interpretation cost. The initial MVP design assumption is a
branching factor of roughly 2–4 candidate actions per state and a solution
depth of roughly 5–30 interpretation actions; the engine records expansion
and frontier statistics so those assumptions can be checked against real
sessions later. With an admissible heuristic, A* provides a least-cost
solution; without one, the result is explicitly not labeled as
optimality-guaranteed. The future Knowledge Base heuristic should be a lower
bound on the remaining interpretation cost—for example, the minimum cost of
covering each still-unexplained interval—so it does not overestimate the true
cost of completing the log.

The initial problem is not adversarial: there is no opponent choosing actions
against IAAIS and no competitive utility to maximize. Minimax and MCTS are
therefore outside this module. They would only become relevant if a later
IAAIS feature modeled an opposing agent or a stochastic game-like decision.

The engine is intentionally domain-neutral. It does not contain exercise
facts or Knowledge Base rules. Later modules can supply those through the
action function and heuristic, while the Planner can reuse the same interface
for action-sequence search.

Run the demonstration in `notebooks/01_search_engine.ipynb` after selecting
the `Python (IAAIS)` kernel. Behavioral tests are in
`tests/test_search_engine.py`.

### Knowledge Base — Chapter 3

The Knowledge Base uses a small Horn-style production-rule formalism. It
stores grounded facts with confidence, status, source, timestamps or
intervals, polarity, and provenance. Forward chaining derives reviewable
facts, while queries distinguish `ENTAILED`, `CONTRADICTED`, `CONFLICTED`, and
`UNKNOWN`.

The module uses an open-world assumption: missing evidence remains unknown.
Explicit negative facts are required to establish contradiction. Explanations
retain the supporting fact IDs and rule names in an indented explanation tree.
`KnowledgeBaseSearchAdapter` connects Knowledge Base evidence to the generic
Search Engine without moving domain rules into the search algorithm. It can
also supply `check_constraints`, `estimate_remaining_cost`, and
`record_search_result` callbacks for the later Planner integration.

Run `notebooks/02_knowledge_base.ipynb` for the facts, rules, open-world
queries, explanations, and cross-module search demonstration. Behavioral tests
are in `tests/test_knowledge_base.py` and
`tests/test_knowledge_base_search_adapter.py`.

### Planner — Chapter 4

The deterministic Planner reads a Knowledge Base snapshot into an immutable
symbolic state, applies grounded STRIPS-style actions, and calls the Search
Engine with a delete-relaxed lower-bound heuristic. Successful and
unsuccessful plan attempts are written back as proposed `plan_result` and
search-run facts. Planner bookkeeping is excluded from later world-state
snapshots so a plan does not accidentally plan from its own history.

Proposed evidence can support a provisional plan, but the result is marked for
review. Conflicting active positive and negative evidence blocks planning until
it is reviewed. `KnowledgeBase.retract_fact` rejects corrected evidence without
deleting its history and invalidates active conclusions derived from that
evidence. `Planner.replan` then generates a new plan from the corrected KB
state and records which earlier plan it replaces. Execution updates reported by
the future Decision Agent remain proposed; the Planner does not control a
person or device and does not silently overwrite confirmed evidence.

The current IAAIS domain is a finite interpretation workflow, so it uses the
deterministic path. `ValueIterationPlanner` is also available as a separate
generic component for future stochastic domains; it returns a policy rather
than pretending that one fixed sequence is valid under probabilistic
outcomes.

Run `notebooks/03_planner.ipynb` for both demonstrations. Tests are in
`tests/test_planner.py` and `tests/test_planner_mdp.py`.

### Uncertainty Module — Chapter 5

The Uncertainty Module keeps probability distributions and evidence histories
separate from deterministic Knowledge Base facts. Categorical observations use
exact Bayes updates; confirmed Knowledge Base facts can condition a belief,
while proposed, unknown, and conflicted evidence is not silently promoted to
certainty. Classifier predictions retain their full distribution.

`UncertaintyModule.expected_utility` supplies expected values to a planner by
weighting each outcome by its posterior probability. A bootstrap particle
filter handles noisy continuous measurements and vector-valued states. Reports
include the full posterior, entropy, information gained from modeled evidence,
source records, and optional Brier, log-loss, and expected calibration error
metrics when labeled outcomes are available.

Run `notebooks/04_uncertainty.ipynb` for the categorical, sensor, planning, and
calibration demonstrations. Tests are in `tests/test_uncertainty.py`,
`tests/test_uncertainty_particle_filter.py`, and
`tests/test_uncertainty_knowledge_base.py`.

### Classifier — Chapter 6

The first supervised task is one exercise/rest label per accelerometer and
gyroscope window. `InertialFeatureEngineer` creates per-axis and vector
magnitude summaries, signal magnitude area, sample rate, duration, and an
optional sensor-profile category. The feature facts are stored as
`sensor_feature(window_id, feature_name, value)` records. The feature extractor
can also read configured fact predicates and statuses from the Knowledge Base.

`Classifier` trains a regularized multiclass logistic-regression model on
explicitly supplied labeled examples or on feature facts paired with confirmed
`exercise_label` facts. Class-weighted training addresses class imbalance.
Macro-F1 is the primary holdout metric because it gives each exercise class
equal weight; accuracy, balanced accuracy, per-class precision and recall, and
a confusion matrix are reported alongside it. The logistic model itself
optimizes class-weighted log loss; macro-F1 is the primary evaluation metric
for future model selection, and no tuning search is claimed yet. Holdout
examples can include recording group IDs, and evaluation rejects overlap with
training groups to reduce window leakage.

Predictions include the complete model probability distribution, a ranked
linear feature-contribution explanation, and a configurable review flag
(initial threshold 0.60). A prediction written to the Knowledge Base remains
`PROPOSED`: the top label is an `exercise_prediction` fact, and each class
probability is a separate `exercise_class_probability` fact. Both retain
timestamps and feature-fact provenance. If supplied, the Uncertainty Module
receives the full distribution as the current belief. These probabilities are
model estimates, not claimed to be calibrated: there is no labeled project
dataset in this repository yet, so no model performance or calibration result
is reported. The threshold is an initial review policy that should be set using
held-out data. The linear contributions explain model scores, not causation.

Run `notebooks/05_classifier.ipynb` for feature engineering, a clearly
illustrative toy training set, holdout evaluation, and the Knowledge Base and
Uncertainty Module handoff. Tests are in `tests/test_classifier.py`.

## WSL setup

From Bash in WSL:

```bash
cd ~/projects/iaais
source iaais_env/bin/activate
```


## Dependencies

Install the direct dependencies into the active environment:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Jupyter

The environment is registered as the `Python (IAAIS)` kernel. In VS Code, open a notebook, use the kernel picker, and select `Python (IAAIS)`. Run `notebooks/00_environment_check.ipynb` to verify the setup.

## Verification

```bash
python -m pytest
python -m pip check
```

## Streamlit

```bash
streamlit run app.py
```

Never commit `.env`. Keep real API keys out of source control and use `.env.example` only as a template.
