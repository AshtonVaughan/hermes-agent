# jev — TypeSafe Jev as Hermes's decision layer

Jev decides, Hermes acts, DeepSeek V4.1 Flash (`deepseek-flash`) writes. Jev returns calibrated
probabilities over closed sets and never writes text, so it handles the small, frequent decisions
that would otherwise cost an LLM round trip or go unmade.

| Where | Jev question | Effect |
|---|---|---|
| `pre_llm_call`, once per turn | Score: how much reasoning does this need? Choice over the skill roster + 3 "is an action wanted" nouls; a rerank of the top 3 with full SKILL.md excerpts | Injects `<skill_relevance>` naming at most one skill (TypeSafe's Hermes cookbook, which more than halved wrong skill loads) |
| `llm_request` middleware, every DeepSeek call in the turn | (uses the turn's reasoning score) | Sets DeepSeek `thinking` off / `reasoning_effort` low / high. Simple turns skip thinking entirely. |
| `pre_tool_call`, gated tools only | Noul: does this call serve what the user asked for? | Below 0.15 the call is blocked with a message to the model. Catches prompt injection (a web page or email telling the agent to run commands or send messages). Never asks the user. |

Everything fails open: no key, a missing SDK, a Jev timeout (4 s) or error means stock Hermes behavior.
DeepSeek's `reasoning_content` echo requirement is safe to toggle per turn because Hermes pads it
on every DeepSeek assistant turn.

## Setup

```bash
hermes plugins enable jev            # installs typesafe-sdk through PM
hermes config set model.provider deepseek
hermes config set model.default deepseek-flash
```

Put `TYPESAFE_API_KEY` and `DEEPSEEK_API_KEY` in `~/.hermes/.env`.

## Settings

`plugins.entries.jev.settings.*` in `config.yaml`: `jev_model`, `timeout`, `skill_suggest`,
`skill_gate`, `skill_fit`, `reasoning_gate`, `reasoning_low`, `reasoning_high`, `action_gate`,
`action_block_below`, `gated_tools`. Defaults are in `plugin.yaml`.

## Inspecting decisions

`/jev [n]` in a session prints the last n decisions. The full log is
`~/.hermes/plugin-data/jev/trace.jsonl`.

## Tests

```bash
scripts/run_tests.sh tests/plugins/test_jev_plugin.py
```
