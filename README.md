# RAP: standalone Tool LLM runtime

`RAP/` is a self-contained copy of the current Tool LLM implementation. The
runtime under `src/` is copied from the parent project, including the traffic
data loader, OSRM route handling, Base forecaster, Normal counterfactual
retrieval, incident retrieval indexes, paired-state evidence, episode memory,
reflection, prompt construction, response parsing, and provider scheduling.

This release intentionally contains no traffic dataset, OSRM cache, model
checkpoint, experiment output, baseline evaluator, visualization code, or API
key. Put those external assets in the locations described below.

## Install

Use Python 3.10 or newer. Install the runtime dependencies in an isolated
environment:

```bash
cd RAP
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The copied model factory includes every model architecture supported by the
current checkpoint loader. A matching PyTorch checkpoint is still required at
run time; checkpoints are deliberately not distributed here.

## External assets

Create the following layout after downloading the prepared dataset and model
artifacts from the project's network-drive location:

```text
RAP/
├── data/processed/<region>_2024/
│   ├── node_order.npy
│   ├── sensor_meta_feature.csv
│   ├── adj_matrix.npy
│   ├── incidents_y2024.csv
│   ├── year_2024/2024_pXX.npy
│   └── ...
├── artifacts_<region>/
│   ├── models/<model>_<region>.pt
│   └── retrieval/                 # generated or copied caches
└── src/llmmanager/providers.json
```

The dataset download URL is distributed separately with the project. Keep the
download outside version control and unpack it under `RAP/data/`; the checked
out repository only reserves `RAP/data/` with `.gitkeep`.

The default region paths are defined in `src/data/regions.py`. Use `--data-dir`
and `--checkpoint` when your local layout differs. OSRM distance and bearing
cache data must be available for the configured impact bands. The default
configuration expects an OSRM service at `http://127.0.0.1:5000`; change this
in `configs/default.json` if a prepared cache or another service is used.

## Provider configuration

`src/llmmanager/providers.json` is a sanitized OpenAI-compatible template. It
contains no credential. Set the key before an LLM run:

```bash
export RAP_LLM_API_KEY='your-api-key'
```

Change `base_url` and `model` in that file for another OpenAI-compatible
provider. The exact `${ENV_VAR}` form is expanded once at process startup and
the resulting provider settings are frozen for the run.

## Run

From any directory, invoke the wrapper; it changes to the RAP root before
loading relative paths:

```bash
python RAP/main.py \
  --region sacramento \
  --data-dir data/processed/sacramento_2024 \
  --checkpoint artifacts_sacramento/models/agcrn_sacramento.pt \
  --output-dir outputs/incident-evaluations/sacramento/agcrn/tool-llm \
  --incident-id 123456 \
  --call-llm
```

For a normal multi-case selection, omit `--incident-id` and use
`--select-count`/`--select-month`. `--incident-workers`, `--device`, and
`--config` expose the same controls as the current project entry point.

Without `--call-llm`, the runtime still prepares Base, Normal, and retrieval
evidence but does not send forecast or reflection requests. LLM output and
case artifacts are written below the requested output directory; retrieval and
episode caches are kept under the corresponding `artifacts_<region>/` paths or
the output directory as configured by the runtime.

## Configuration

`configs/default.json` preserves the current context switches and retrieval
groups. In particular, `base_prediction` controls loading the foundation model
and all Base-dependent retrieval; `normal_reference` controls Normal retrieval
and all Normal-dependent evidence. `phased_prediction` selects the current
structured phase output contract. The three incident retrieval groups and
their top-k values are configured independently under `retrieval`.

## Minimal verification

These checks do not require data, a checkpoint, or a provider key:

```bash
python RAP/main.py --help
python -m compileall -q RAP/src RAP/main.py
```

The first command verifies the standalone CLI can be imported without loading
heavy model dependencies. The second verifies every copied Python module is
syntactically complete. A real forecast requires the external assets and
dependencies above; `--help` cannot validate those external services.

## Repository contents

```text
RAP/
├── main.py                 # standalone entry point
├── configs/default.json    # current runtime configuration
├── configs/models/         # supported checkpoint model configurations
├── src/                    # copied Tool LLM runtime and model implementations
├── data/                   # empty dataset mount point
├── artifacts/              # empty generated-cache mount point
├── outputs/                # empty generated-output mount point
├── requirements.txt
└── README.md
```
