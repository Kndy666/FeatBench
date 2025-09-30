# FeatBench

FeatBench is an end-to-end benchmark pipeline that turns real GitHub feature releases into executable evaluation tasks for software agents. It fills the gap between traditional code-completion benchmarks and emerging "vibe coding" workflows where non-experts rely on large language models (LLMs) to describe functionality in natural language and let agents implement it autonomously. FeatBench covers data collection, dataset construction, containerized environment setup, and automated test execution so you can measure how well agents implement feature-level changes.


## Benchmark Highlights

- **Pure natural-language prompts** – task inputs contain only abstract user-facing descriptions with no code snippets or signature hints, mirroring vibe coding interactions.
- **Release-grounded corpus** – each instance originates from a curated GitHub release and pull-request history, yielding high-signal requirements and verified reference patches.
- **Rigorous, evolving pipeline** – a multi-stage, fully automated collection system applies quality filters, mitigates data contamination, and can roll forward continuously as new releases ship.
- **Comprehensive regression checks** – Fail-to-Pass (F2P) and Pass-to-Pass (P2P) pytest selections ensure both new behaviour and legacy functionality are validated.
- **Diverse domains** – 27 actively maintained repositories spanning AI/ML, DevOps, web platforms, and productivity tools provide broad coverage of real-world tech stacks.

## Repository Layout

```
FeatBench/
├── data/                         # Example processed evaluation outputs
├── data_collect/                 # GitHub mining and release/PR analysis pipeline
├── docker_agent/                 # Docker-based execution, patch application, and scoring
├── requirements.txt              # Python dependencies shared across modules
└── README.md
```

## Prerequisites

- Python 3.10 or later (3.11 recommended to match the configured containers).
- Docker Engine 24+ with the NVIDIA runtime enabled if you want GPU acceleration (the default container spec requests GPUs).
- Recent Git and curl installations for repository cloning inside containers.
- Access tokens:
  - A GitHub personal access token with `repo` and `read:org` permissions.
  - An LLM provider key (DeepSeek/OpenAI-compatible) for PR summarisation.
- Optional but recommended: [uv](https://github.com/astral-sh/uv) for faster, deterministic Python installs inside containers.

## Installation

1. **Clone the repository and create an environment**

	```bash
	git clone https://github.com/Kndy666/FeatBench.git
	cd FeatBench
	python -m venv .venv
	source .venv/bin/activate
	python -m pip install --upgrade pip
	```

2. **Install Python dependencies**

	```bash
	pip install -r requirements.txt
	```

	The requirements file covers both the data collection tools and the Docker agent controller. Inside containers FeatBench will additionally install project-specific dependencies via `uv` or `pip` as directed by the prompts.

## Configure Access Tokens and Defaults

### Data collection (`data_collect/`)

1. Copy the template and fill in your credentials:

	```bash
	cp data_collect/config.toml.template data_collect/config.toml
	```

2. Edit the new `config.toml` and replace the placeholder GitHub and OpenAI-compatible keys (consider loading them from environment variables instead of committing secrets).

3. Adjust crawling parameters if needed:
	- `crawl_mode` controls whether to analyse star-ranked repositories or a curated list in `crawl.json`.
	- `release_collector` thresholds (stars, release count, etc.) gate which repositories enter the pipeline.

### Docker agent (`docker_agent/`)

1. Copy the template config provided in `docker_agent/config.toml.template` and edit paths, proxy settings, and timeouts to match your environment.
2. Set `paths.analysis_file` to the dataset JSON generated in the previous stage.
3. Review `docker.base_image` and the Dockerfile template if you cannot provide an NVIDIA-enabled runtime; you can comment out the GPU-related settings in `docker_setup.CacheManager.common_container_config` when running on CPU-only hosts.

## Running the Pipeline

### 1. Collect release-driven tasks

```bash
python -m data_collect.main --crawl-mode specified
```

This command orchestrates four stages:

1. Collect candidate repositories (`collect_repositories`).
2. Analyse releases for new features and improvements (`analyze_releases`).
3. Enrich promising releases with PR-level diffs and LLM-authored task descriptions (`enhance_with_pr_analysis`).
4. Persist the combined output to `swebench-live/final_analysis_results.json` (path configurable via `config.toml`).

Intermediate caches live beside the output directory so re-runs can reuse prior work. Use `--no-cache` to force fresh API calls.

### 2. Transform releases into evaluation specs

Use `docker_agent/dataset_transformation.py` to generate the agent-facing JSON structure:

```bash
python docker_agent/dataset_transformation.py \
  --input swebench-live/final_analysis_results.json \
  --output docker_agent/analysis_results.json
```

The resulting file groups patches by repository, separates test changes (`test_patch`) from source patches (`patch`), records metadata such as `instance_id`, and becomes the input for the Docker runner.

### 3. Prepare Docker configs and cached environments

The first invocation of the Docker agent performs two stages per repository:

1. **File discovery** – run trae-agent locally to catalogue setup files and infer the recommended Python version. Outputs land in `docker_agent/swap/<repo>/`.
2. **Environment setup** – create a container, install dependencies using the recommended tooling (`uv` preferred, `pip` fallback), and save the configured container as a cached image for faster retries.

You can trigger the bootstrap process with:

```bash
python docker_agent/run.py
```

Use `--test-only` to skip recomputing environments and focus on already-processed specs.

### 4. Evaluate agents and patch-based baselines

After environments are prepared, run the evaluator to measure each agent:

```bash
python docker_agent/evaluate.py --agent-names trae-agent agentless --max-instances 20
```

Key behaviours:

- Containers are provisioned via `DockerEnvironmentManager`; active container IDs are tracked so Ctrl+C triggers a clean shutdown.
- Each spec applies the historical test patch, then runs targeted pytest selections derived from `FAIL_TO_PASS` and `PASS_TO_PASS` lists.
- Agent runs, logs, and test results are appended to `docker_agent/data/processed_evaluation_results_*.json` files for further analysis.

## Output Artefacts

- `data/processed_evaluation_results_*.json` – curated evaluation summaries for different agent/model combinations.
- `docker_agent/logs/test_logs.json` – pytest logs before/after applying generated patches.
- `docker_agent/swap/` – transient working tree used for cloning repositories, storing prompts, and caching recommended Python versions. This directory can be purged to reclaim disk space.

## Development Tips

- The command executors inject terminal dimensions and proxy settings from `docker_agent/command_executor.py`. Adjust `docker_environment` if your infrastructure requires different proxies.
- All long-running docker operations respect the `docker.timeout` setting in `docker_agent/config.toml`.
- To run without GPUs, edit `CacheManager.common_container_config` to remove the `device_requests` section and switch the runtime to `runc`.
- Enable verbose logging by setting `logging.level = "DEBUG"` in the respective `config.toml` file.

## Troubleshooting

- **API rate limits** – the data collection stage makes extensive GitHub API calls; prefer tokens with elevated rate limits and consider adjusting the sleep intervals in `pr_analyzer.py` if you still hit secondary limits.
- **Trae-agent failures** – ensure the `trae-agent` repository can be cloned from within containers.
- **Patch application conflicts** – the `PatchAnalyzer` records which files fail to apply; inspect `docker_agent/logs/evaluation.log` for mismatches and rerun with `--test-only` once resolved.
- **Timeouts** – pytest commands default to `--timeout=5` and may need relaxation for larger projects. Modify the prompt template in `docker_agent/config.toml` if longer tests are acceptable.

> **Note:** This is a temporary release and the code may contain various bugs. We will provide a more stable version as soon as possible to facilitate future community use.