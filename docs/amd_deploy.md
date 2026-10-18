# AMD Developer Cloud AI deployment

This guide deploys the optional inference service in `server.py`. AMD ROCm/PyTorch packages stay on the AMD host. Matching uses AMD embeddings when the service is healthy; otherwise it scores available structured details such as distance, availability, language, reliability, and explicit constraints. It does not load an embedding model on the Flask CPU or substitute keyword similarity.

## Matching backends at a glance

| Backend | When it is used | What to deploy |
| --- | --- | --- |
| AMD service (`server.py`) | `AI_BASE_URL`/`AI_API_KEY` set and `/health` responds | This guide |
| Manual component scoring | AMD embeddings unavailable | No model installation; uses structured request and profile details |

Manual scores omit semantic similarity and equally weight the available non-semantic components. If no such components or explicit constraints are available, no match score is produced.

## Before starting

1. In the AMD Developer Cloud account, check the instance instructions for your allocation and record the exact GPU model, VRAM, Linux distribution, SSH host, username, and SSH port. Those account-specific values were not supplied, so this guide does not invent an instance-creation or login command.
2. Verify the GPU and Linux distribution in AMD's current [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html). Pick the matching official ROCm PyTorch Docker tag from [AMD's current PyTorch install guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html). GPU and ROCm compatibility changes; do not assume the example tag below suits your allocation.
3. Check the current Hugging Face model card and license for [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it), including whether the repo is gated and whether an access agreement must be accepted. The card currently identifies an Apache 2.0 license. The card documents `AutoProcessor` and `AutoModelForMultimodalLM`; verify those names and the model ID against the live card before launching. Verify available VRAM and a successful load before using the 31B model. If it does not fit, select a smaller supported model from the live Gemma collection and set its exact repo ID in `MODEL_ID`.
4. The optional semantic matching endpoint uses [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), whose current card lists 50 languages and Apache 2.0. Verify the live model card and license before setting `EMBEDDING_MODEL_ID`.

## ROCm PyTorch environment

The recommended AMD route is their prebuilt PyTorch container. The tag below is an example shown in AMD's current install documentation, not a compatibility promise for an unknown GPU. **VERIFY the tag, ROCm release, Python version, OS, and GPU compatibility on AMD's current pages before running either command.**

Transfer only `server.py` and `server_requirements.txt` to a working directory on the AMD host using the account's approved private method, or use the complete project checkout if permitted. Do not copy the Windows `.env`, Supabase keys, or local database to the model server. Persist the Hugging Face cache on the AMD host so model weights and the login token survive container recreation.

```bash
# Create a private host-side cache directory before starting the container.
mkdir -p "$HOME/.cache/huggingface"

# VERIFY image tag against AMD's current PyTorch install guide and your instance GPU/OS.
docker pull rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0

# VERIFY Docker/device access requirements with the AMD instance instructions.
# The host-side published port is bound to loopback, not a public interface.
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --ipc=host --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD:/workspace" -w /workspace \
  rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0 bash
```

Inside the container, confirm PyTorch sees the accelerator, then install only the server-side dependencies. **VERIFY the test command against AMD's current PyTorch/ROCm instructions.**

```bash
# VERIFY PyTorch's ROCm support and detected device on this image.
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")'

# Run from the mounted project directory; this does not replace the container's ROCm PyTorch.
python -m pip install -r server_requirements.txt
```

PyTorch on ROCm exposes AMD devices through the `torch.cuda` API. `server.py` deliberately loads with `device_map="cuda"` as requested. If the availability test fails, stop and resolve the AMD driver, ROCm image, GPU support, or container access before loading a model.

## Hugging Face access and model

If the selected model repository requires authentication or an accepted license, complete that step on the Hugging Face model page using the account that will download it. Use a read token; do not paste the token into source files or shell commands. The current documented Hub login command is below; **VERIFY it against the live Hugging Face CLI documentation**.

```bash
# VERIFY the current CLI command at https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#hf-auth-login
hf auth login
hf auth whoami
```

Set the chosen model ID and generate a separate API key for Flask-to-inference requests. Store the same key securely in the Flask app's local `.env`; do not commit it or publish it in the browser.

```bash
# VERIFY the model ID and loading API against its current Hugging Face model card.
export MODEL_ID=google/gemma-4-31B-it

# VERIFY the model ID and license on its current Hugging Face card.
export EMBEDDING_MODEL_ID=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Generate a key locally in this shell; do not print or commit its value.
export AI_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

## Start and keep the port private

Inside the Docker container, start Uvicorn bound to the container interface. The Docker command above publishes it only on the AMD host's loopback interface. **VERIFY this launch command against the installed Uvicorn version. Do not publish port 8000 to a public interface or open it in a public firewall rule.**

```bash
# VERIFY command and port mapping for your selected AMD instance/container.
uvicorn server:app --host 0.0.0.0 --port 8000
```

From the Windows machine running Flask, open an SSH local forward using the host, user, port, and key information supplied by the AMD account. Keep the tunnel terminal running. **VERIFY the SSH destination and options against the instructions for your specific allocation.**

```powershell
# VERIFY placeholders and SSH options with the AMD instance instructions.
ssh -N -L 8000:127.0.0.1:8000 <user>@<amd-host>
```

Set these values in the Flask app's ignored `.env` file. `AI_API_KEY` must match the value set on the AMD host. The local base URL points through the SSH tunnel; do not put the AMD host's public address here.

```dotenv
AI_BASE_URL=http://127.0.0.1:8000
AI_API_KEY=<same-private-key-as-the-server>
AI_TIMEOUT_SECONDS=20
```

## Smoke test

With the server and SSH tunnel running, run these commands from PowerShell. Use synthetic text only. **VERIFY `curl.exe` options on your Windows installation.**

```powershell
# VERIFY local tunnel and service health; this endpoint reports readiness and model ID.
curl.exe -f http://127.0.0.1:8000/health

# VERIFY authenticated generation using a non-sensitive test prompt.
curl.exe -f -H "Content-Type: application/json" -H "x-api-key: $env:AI_API_KEY" --data-binary '{"text":"Write one sentence offering help with a ride in Genova.","max_new_tokens":64}' http://127.0.0.1:8000/generate
```

A successful generation returns `text`, `latency_s`, `tokens`, and `model`. If the remote server is unavailable, the Flask client retries once and switches to the local keyword/rule fallback. The site header reports which backend currently answers `/health`.

## Official references

- [AMD ROCm PyTorch install guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
- [AMD ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [Hugging Face Gemma 4 model card](https://huggingface.co/google/gemma-4-31B-it)
- [Transformers Gemma 4 documentation](https://huggingface.co/docs/transformers/main/en/model_doc/gemma4)
- [Hugging Face CLI login documentation](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#hf-auth-login)
