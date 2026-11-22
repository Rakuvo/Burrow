<div align="center">

# Rakuvo

### A reliability gateway for AI agents.

Change one line, the base URL. Six layers go to work between your agent and its model.

[![License](https://img.shields.io/badge/license-Apache--2.0-1f7a5a)](LICENSE)
[![Go](https://img.shields.io/badge/built%20with-Go-00ADD8)](go.mod)
[![OpenAI compatible](https://img.shields.io/badge/API-OpenAI%20compatible-black)](#use-it)

[Quick start](#quick-start) &middot; [How it works](#how-it-works) &middot; [The six layers](#the-six-layers) &middot; [Providers](#providers) &middot; [Evidence](#what-the-evidence-says)

</div>

---

```diff
  const client = new OpenAI({
-   baseURL: "https://api.openai.com/v1",
-   apiKey: process.env.OPENAI_API_KEY,
+   baseURL: "http://localhost:8080/v1",     // Rakuvo
+   apiKey: process.env.RAKUVO_TOKEN,
  });
```

That is the whole integration. Your agent code, your tools and your prompts stay as they are.

## Why this exists

Teams that build agents keep running into the same four failures:

| What happens | What it looks like |
|---|---|
| **The model answers from memory** | It should have called the tool. It guessed, or called it with a broken argument. |
| **The agent loops** | It repeats the same call, with the same result, dozens of times. |
| **A long job runs out of room** | Old tool output fills the context window until the run fails. |
| **The agent forgets** | A user said where they live last week. Today it asks again, or uses the old answer. |

Fixing these inside every agent means rewriting every agent. Rakuvo fixes them once, in a layer the agent never sees, and **measures each fix per model**, because a fix that doubles accuracy on one model can do nothing on another.

> One incident, reported by several outlets: in July 2025 a coding agent deleted a company's production database during a code freeze it had been told about, then gave a wrong account of what it had done. An instruction in a prompt is not a control, and a model's report of its own actions is not evidence. Rakuvo's layers act on what the agent actually does.

## Quick start

You need Go and a key from any [supported provider](#providers).

```bash
git clone https://github.com/Rakuvo/Burrow.git
cd Burrow
go build -o rakuvo ./cmd/rakuvo

export OPENROUTER_API_KEY=...        # or OPENAI_API_KEY, DEEPSEEK_API_KEY, ...
export RAKUVO_TOKEN=$(openssl rand -hex 24)   # the key your agent will send

./rakuvo serve -provider openrouter -listen 127.0.0.1:8080 -token-env RAKUVO_TOKEN \
  -guard -tool-policy-auto -loop-guard
```

Check it:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $RAKUVO_TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-4o-mini","messages":[{"role":"user","content":"Say hi"}]}'
```

Then point any OpenAI-compatible client at `http://127.0.0.1:8080/v1`:

<details open>
<summary><b>JavaScript</b></summary>

```js
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://127.0.0.1:8080/v1", apiKey: process.env.RAKUVO_TOKEN });
```
</details>

<details>
<summary><b>Python</b></summary>

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key=os.environ["RAKUVO_TOKEN"])
```
</details>

Name your agent with a header and every call shows up under it in the dashboard: `X-Rakuvo-Agent: my-agent`. Send a `user` field and Rakuvo remembers that user across chats.

## How it works

```
  your agent ──► Rakuvo gateway ──► model provider
                   │
                   ├─ before: layers edit the request   (in order)
                   ├─ the provider call
                   └─ after:  layers check the answer   (in reverse order)
                              and may ask for a retry
```

- **One pipeline, many layers.** Each layer has a `Before` and an `After` step. After runs in reverse, so the layer closest to the model sees the answer first.
- **The engine owns retries.** A layer never retries on its own. It asks, and the engine re-runs the pipeline on a fresh copy of the original request, up to a cap. Two layers can't fight each other, and a retry never applies a prompt edit twice.
- **Metadata only.** The trace records tokens, latency, retries and what each layer did. It does not record prompts or answers.
- **OpenAI format in, any provider out.** Agents speak the OpenAI API. Claude is reached through a native adapter that translates each call.

## The six layers

Switch on only what helps your model. Every layer is off until you ask for it, except the tool-call guard.

<table>
<tr><th>Layer</th><th>What it does</th><th>Turn it on</th></tr>
<tr><td><b>Prompt</b></td><td>Adds one short instruction so models use their tools instead of answering from memory. <code>auto</code> adds it only for models where it was measured to help.</td><td><code>-tool-policy-auto</code> or <code>-tool-policy</code></td></tr>
<tr><td><b>Tools</b></td><td>Checks every tool call against its schema and repairs small mistakes (a markdown fence, a trailing comma). An optional small judge model can veto a call that makes no sense.</td><td><code>-guard</code> (on by default), <code>-verify-model</code></td></tr>
<tr><td><b>Loop guard</b></td><td>Spots an agent repeating itself, nudges it, then stops it with an honest report. Healthy repetition such as polling is allowed.</td><td><code>-loop-guard</code>, <code>-loop-allow-repeat</code></td></tr>
<tr><td><b>Context</b></td><td>Shrinks old tool results to short notes the agent can fetch again, so long jobs fit. Can also prune tool definitions and summarise old turns.</td><td><code>-context-keep-observations</code>, <code>-context-mask-within-turn</code></td></tr>
<tr><td><b>Memory</b></td><td>A graph of what each user said, with dates. A newer fact replaces an old one, and the old one stays as history marked no longer true.</td><td><code>-memory-dir</code>, <code>-memory-model</code>, <code>-memory-embed-model</code></td></tr>
<tr><td><b>Model policy</b></td><td>Optional. Tries a cheap model first and escalates to a strong one when the cheap one fails.</td><td><code>-model-cheap</code>, <code>-model-strong</code></td></tr>
</table>

A fuller setup:

```bash
./rakuvo serve -provider openrouter -token-env RAKUVO_TOKEN \
  -tool-policy-auto -guard -loop-guard -loop-max-tool-calls 15 \
  -context-keep-observations 2 -context-mask-within-turn \
  -memory-dir ./memory -memory-model openai/gpt-4o-mini \
  -memory-embed-model openai/text-embedding-3-small \
  -trace ./trace.jsonl
```

> **Memory needs an embedding model for real recall.** Without `-memory-embed-model`, search matches words only, so "which city am I in?" will not find "lives in Pune". `./rakuvo serve -h` lists every flag.

## Providers

`-provider <name>` sets the base URL, the key variable and the token field. Your agent keeps speaking the OpenAI format.

| `-provider` | Key variable | Notes |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude via its native Messages API. Experimental: unit-tested, not yet run against the live API. |
| `deepseek` | `DEEPSEEK_API_KEY` | |
| `openrouter` | `OPENROUTER_API_KEY` | Model names look like `vendor/model`. |
| `gemini`, `groq`, `together`, `fireworks`, `mistral`, `xai` | `GEMINI_API_KEY`, `GROQ_API_KEY`, ... | OpenAI-compatible endpoints. |
| `ollama`, `vllm`, `lmstudio` | none | Open-source models on your own machine. |
| `custom` | `-upstream-key-env` | Any OpenAI-compatible server, with `-upstream <url>`. |

`-upstream`, `-upstream-key-env` and `-max-tokens-field` override a preset.

## What the evidence says

Every number below is the same questions run with and without Rakuvo, on a public benchmark and its official scorer. Differences are in percentage points, paired by question, with 95% intervals. Settings were written down before each run, tuned on a development split, and scored once on held-out data.

| Layer | Test | Model | Result |
|---|---|---|---|
| Prompt | BFCL multi-turn tool calling, 400 entries | gpt-5-mini | 19.2% to 35.5% (**+16.2**, interval +11.2 to +21.0). Cost per correct answer fell 44%. |
| Prompt | BFCL multi-turn, 800 entries | gpt-5-nano | **+6.4** (interval +3.2 to +9.5) |
| Tools | BFCL single-turn, judge and guard | gpt-5-nano | **+4.4** (interval +2.9 to +5.8) |
| Memory | LoCoMo long-term memory, 1,155 held-out questions | gpt-4o-mini reader | **68.9%** against 58.4% for embedding search. It also beat putting the whole conversation in the prompt (63.9%) while showing 2.6% of the text. |
| Memory | Same questions against open-source Mem0 | gpt-4o-mini reader | **64.2%** against 60.3% |
| Loop guard | 12 fresh scenarios written before the run | gpt-5-nano | Tool calls on impossible tasks fell **4.8 to 2.7**. On healthy tasks it never nudged or stopped the agent. |
| Context | A job reading about 240K tokens on a 128K model | five runs per side | 3 of 5 finished with the layer, none without. A small demonstration, not a statistic. |

**How to read it:** Rakuvo is a credible reliability layer for tool calling, stuck agents and conversational memory. It is strongest on models that fail in the specific ways the layers target, and neutral on models that do not. It is not a switch that improves every agent.

## Run it your way

- **Self-hosted** (this repo): one program in your own network, so prompts never leave it.
- **Hosted**: if you would rather not run it, we can run it for you, with a dashboard for every call, per-agent cost, what each layer did, and a memory graph. White-label hosting is available. Get in touch through [rakuvo.com](https://rakuvo.com).

## Contributing

Issues and pull requests are welcome. A new layer must come with a way to measure it: what it should change, on which benchmark, for which model, and a result that does not depend on the run you tuned it on. Results that find nothing are as welcome as the ones that work.

```bash
go test ./...
```

## License

Apache-2.0. See [LICENSE](LICENSE).

---

<div align="center">

[rakuvo.com](https://rakuvo.com)

</div>
