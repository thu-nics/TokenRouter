<div align="center">
  <img src="resource/logo.png" alt="TokenRouter Logo" width="100"/>

  <h1>TokenRouter</h1>
  <h3>Efficient Serving System for Token-Level LLM Routing</h3>

  <p>
    <a href="https://fuvty.github.io/thinking_yard_project_page/projects/tokenrouter/">🌐 <b>Project Page</b></a> •
    <a href="https://arxiv.org/abs/2610.12242">📑 <b>Paper</b></a> •
    <a href="https://huggingface.co/papers/2610.12242">🤗 <b>HuggingFace</b></a>
  </p>
</div>

TokenRouter is an efficient and developer-friendly serving system for token-level LLM routing. You describe routing logic through three interfaces: `route()`, `send()`, and `receive()`, TokenRouter efficiently serves it for you.
Across diverse routing algorithms, workloads, and model pairs, TokenRouter achieves **2.01–64.15× higher decoding throughput** than existing systems.

Feel free to star the repo or cite the paper if you find it interesting.

```bibtex
@article{fu2026tokenrouter,
    title={TokenRouter: Efficient Serving System for Token-Level LLM Routing},
    author={Tianyu Fu and Tengxuan Liu and Ruoxi Wang and Yixin Dong and Yi Ge and Yichen You and Yu Wang},
    journal={arXiv preprint arXiv:2610.12242},
    year={2026},
}
```

## News

[2026/10] 📑 Our [paper](https://huggingface.co/papers/2610.12242) is now available on arXiv. Explore the [project page](https://fuvty.github.io/thinking_yard_project_page/projects/tokenrouter/) for interactive explanations and results.

[2026/09] 🎉 Our paper is accepted by **NeurIPS 2026**!

## Environment Setup

```bash
conda create -n tokenrouter python=3.10
conda activate tokenrouter
pip install -e .
```

## Quick Start

TokenRouter is configured through a single YAML file. Example configs are provided under [config/](config/), and runnable usage examples are provided under [examples/](examples/).

### Launch a TokenRouter server

Launch a TokenRouter server with a routing config:

```bash
python -m tokenrouter.launch_server --config_path config/r2r/Qwen3-0.6B_Qwen3-32B.yaml
```

To place one model on each node, keep both model definitions in the shared
config, add `system.peer_endpoints`, and select the model launched by each
process. Replace `<QUICK_NODE_IP>` and `<REFERENCE_NODE_IP>` with the reachable IP addresses of the corresponding nodes

```yaml
system:
  entry_model: quick
  peer_endpoints:
    quick:
      bind: "tcp://0.0.0.0:30001"
      advertise: "tcp://<QUICK_NODE_IP>:30001"
    reference:
      bind: "tcp://0.0.0.0:30001"
      advertise: "tcp://<REFERENCE_NODE_IP>:30001"
```

```bash
# On quick node
python -m tokenrouter.launch_server --config_path config.yaml --local_model quick

# On reference node
python -m tokenrouter.launch_server --config_path config.yaml --local_model reference
```

Once the server is running, send a chat completion request with either the OpenAI client example or a plain `requests` example:

```bash
python examples/openai_chat_completion.py
python examples/requests_chat_completion.py
```

### Use TokenRouter as Engine in Python

Use `TokenRoutingEngine` when you want to run TokenRouter directly from Python without starting a separate HTTP server:

```python
from tokenrouter.entrypoints.engine import TokenRoutingEngine

with TokenRoutingEngine("config/r2r/Qwen3-0.6B_Qwen3-32B.yaml") as engine:
    response = engine.generate(
        prompt="<|im_start|>user\nSay hello in one short sentence.<|im_end|>\n<|im_start|>assistant\n",
        sampling_params={"temperature": 0, "max_new_tokens": 256},
    )
    print(response["text"])
```

See [examples/engine_generate.py](examples/engine_generate.py) for a complete example:

```bash
python examples/engine_generate.py
```

## Reproducing Our Results

Select one config and one workload before launching the server.

| Method | Config path |
| --- | --- |
| R2R | [config/r2r/Qwen3-0.6B_Qwen3-32B.yaml](config/r2r/Qwen3-0.6B_Qwen3-32B.yaml) |
| CITER | [config/citer/Qwen3-0.6B_Qwen3-32B.yaml](config/citer/Qwen3-0.6B_Qwen3-32B.yaml) |
| Co-LLM | [config/collm/Qwen3-0.6B_Qwen3-32B.yaml](config/collm/Qwen3-0.6B_Qwen3-32B.yaml) |
| R-Stitch | [config/rstitch/Qwen3-0.6B_Qwen3-32B.yaml](config/rstitch/Qwen3-0.6B_Qwen3-32B.yaml) |
| ME | [config/me/Qwen3-0.6B_Qwen3-8B_Qwen3-32B.yaml](config/me/Qwen3-0.6B_Qwen3-8B_Qwen3-32B.yaml) |

| Workload | Dataset path |
| --- | --- |
| AIME | [evaluation/datasets/aime.json](evaluation/datasets/aime.json) |
| AIME 8K output budget | [evaluation/datasets/aime_8k.json](evaluation/datasets/aime_8k.json) |
| SWE-smith multi-round trajectories | [evaluation/datasets/swesmith_traj_multi_round.json](evaluation/datasets/swesmith_traj_multi_round.json) |

### 1. Launch CUDA MPS

If multiple models share the same physical GPU, CUDA MPS is recommended for reproducing overlapping serving behavior and improving concurrent scheduling performance.

```bash
mkdir -p tmp/mps_${METHOD}_${PORT}

export CUDA_MPS_PIPE_DIRECTORY=$PWD/tmp/mps_${METHOD}_${PORT}/pipe
export CUDA_MPS_LOG_DIRECTORY=$PWD/tmp/mps_${METHOD}_${PORT}/log

nvidia-cuda-mps-control -d
```

### 2. Launch Server

Start the TokenRouter server with the selected routing config.

```bash
python -m tokenrouter.launch_server --config_path $CONFIG_PATH
```

### 3. Run Throughput Benchmark

Run the workload on the server and write per-request measurements to the output directory.

```bash
bash evaluation/test_throughput.sh \
  $CONCURRENCY \
  $PORT \
  $NUM_INPUTS \
  $REPEATS_PER_INPUT \
  $MAX_TOKENS \
  $OUTPUT_DIR \
  $DATASET_PATH
```

The results will be written to `${OUTPUT_DIR}/result.csv`.

Use `evaluation/report.py` to summarize an existing result CSV without rerunning the benchmark.

```bash
python evaluation/report.py --result ${OUTPUT_DIR}/result.csv
```

## Supported Token-level Routing Algorithms

| Method | Scheduler Classes |
| --- | --- |
| R2R | `r2r_slm` + `r2r_llm` |
| CITER | `citer_slm` + `citer_llm` |
| Co-LLM | `collm_base` + `collm_assist` |
| R-Stitch | `rstitch_slm` + `rstitch_llm` |
| GlimpRouter | `glimprouter_slm` + `glimprouter_llm` |
| ME | `random_routing` on `small` + `medium` + `large` |
| Query-level routing | `query_level_routing` |
| Random baseline | `random_routing` |

## Understanding the Code

Code structure and instructions for adding custom routing methods.

<details>
<summary><b>Expand to see</b></summary>

### Code structure

- `tokenrouter/`: The main serving package.
  - `entrypoints/`: HTTP serving, OpenAI-compatible chat completions, and the Python engine.
  - `managers/`: Scheduling, request handoff, tokenization, and detokenization.
  - `managers/schedulers/`: Routing schedulers and the scheduler registry.
  - `algorithms/`: Method-specific routing logic and helpers.
  - `model_executor/`: Model execution and CUDA graph support.
- `config/`: YAML configurations for routing methods and supported model-level arguments.
- `examples/`: Server clients and Python engine examples.
- `evaluation/`: Workloads, throughput benchmarks, reporting, and delayed-batching optimization.

### Implement a new token-level routing method

TokenRouter represents each cooperating model with its own scheduler class. A new routing method usually adds one scheduler for each model. Every scheduler should subclass `BaseTokenRoutingScheduler`, defined in [tokenrouter/managers/schedulers/base_scheduler.py](tokenrouter/managers/schedulers/base_scheduler.py). Place scheduler classes in [tokenrouter/managers/schedulers/](tokenrouter/managers/schedulers/) and keep method-specific helper logic under [tokenrouter/algorithms/](tokenrouter/algorithms/). For each scheduler class, implement three key interfaces: `route()`, `send()`, and `receive()`.

#### 1. Implement `route()`

`route(batch, result)` is called after each forward pass and returns one decision per request: a list of model names or a 1-D `torch.Tensor` with shape `(batch.batch_size(),)`. Each element corresponds to the request at the same index in `batch.reqs`.

Return one routing decision per request:

- Return a model name to route explicitly, for example `"small"`,
  `"medium"`, or `"large"`. Returning the current model name continues
  decoding locally.
- For existing two-model schedulers, `True` / `1` still delegates to the
  only peer and `False` / `0` still decodes locally.

#### 2. Implement `send()`

`send(req, destination)` builds the `PeerReq` message sent to an addressed
scheduler. Use `PeerReq.construct_from_req(...)` to convert the local `Req`
into a peer message, passing the source scheduler name, destination scheduler
name, and the token ids that should be transferred:

```python
current_token_ids = req.origin_input_ids + req.output_ids[:-1]
req_to_send = PeerReq.construct_from_req(
    req=req,
    source=self.server_args.server_name,
    destination=destination,
    current_token_ids=current_token_ids,
    entry_model=self.server_args.entry_model_name,
    handoff_seq=req.handoff_seq,
)
```

The default `send()` implementation uses `req.origin_input_ids + req.output_ids` as `current_token_ids`, so the peer receives the prompt plus all generated tokens currently owned by this scheduler.

#### 3. Implement `receive()`

`receive(msgs)` processes `PeerReq` messages from the peer scheduler. By default, `receive()` accepts all tokens sent by the peer. Override `receive()` when the method needs custom preprocessing before applying peer tokens.

#### 4. Register Scheduler

Register the scheduler in [tokenrouter/managers/schedulers/__init__.py](tokenrouter/managers/schedulers/__init__.py) by adding it to `SCHEDULER_REGISTRY`. The registry key becomes the YAML config value for `scheduler_class`.

#### 5. Optimize Delayed-batching Threshold

The delayed-batching threshold can be optimized using [evaluation/calculate_delayed_batching.py](evaluation/calculate_delayed_batching.py). For example, to optimize the delayed-batching threshold of R2R on Qwen3-0.6B and Qwen3-32B, run the following command:

```bash
python evaluation/calculate_delayed_batching.py \
        --M 2 --N $CONCURRENCY --latencies 6.0 27.9 \
        --P '[[0.35, 0.65], [1, 0]]' \
        --C '[[1, 0], [1, 0]]' \
        --delta 0.1
```

#### 6. Add Config

Each cooperating model is configured independently inside the YAML config. TokenRouter extends `sglang.srt.server_args.ServerArgs` with additional routing-specific fields defined in [tokenrouter/server_args.py](tokenrouter/server_args.py). All the supported model-level configuration keys are listed in [config/supported_server_args.yaml](config/supported_server_args.yaml).

</details>

## Related Projects

Explore more efficient LLM projects from us:

<table style="border: none; border-collapse: collapse;" align="center">
<tr>
<td align="center" valign="top" width="20%" style="border: none; border-right: 1px solid rgba(128, 128, 128, 0.3); padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/C2C">
<img src="https://raw.githubusercontent.com/thu-nics/C2C/main/resource/logo.png" alt="Cache-to-Cache Logo" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/C2C"><b>C2C</b></a>
<br/><sub>Direct communication through KV-caches</sub>
</td>
<td align="center" valign="top" width="20%" style="border: none; border-right: 1px solid rgba(128, 128, 128, 0.3); padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/R2R">
<img src="https://raw.githubusercontent.com/thu-nics/R2R/main/resource/logo.png" alt="R2R Logo" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/R2R"><b>R2R</b></a>
<br/><sub>Token-level routing for reasoning LLMs</sub>
</td>
<td align="center" valign="top" width="20%" style="border: none; border-right: 1px solid rgba(128, 128, 128, 0.3); padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/TaH">
<img src="https://raw.githubusercontent.com/thu-nics/TaH/main/resource/logo.png" alt="TaH Logo" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/TaH"><b>TaH</b></a>
<br/><sub>Selective latent thinking for reasoning LLMs</sub>
</td>
<td align="center" valign="top" width="20%" style="border: none; border-right: 1px solid rgba(128, 128, 128, 0.3); padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/FrameFusion">
<img src="https://raw.githubusercontent.com/thu-nics/FrameFusion/main/example/image/logo.png" alt="FrameFusion Logo" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/FrameFusion"><b>FrF</b></a>
<br/><sub>Efficient video token reduction for LVLMs</sub>
</td>
<td align="center" valign="top" width="20%" style="border: none; padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/MoA">
<img src="https://raw.githubusercontent.com/thu-nics/MoA/master/resource/logo.png" alt="MoA Logo" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/MoA"><b>MoA</b></a>
<br/><sub>Mixture of sparse attention for LLMs</sub>
</td>
</tr>
</table>
