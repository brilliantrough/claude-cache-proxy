# OpenRouter Model Map JSON Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow OpenRouter model aliases such as `claude-opus-4-7` to be maintained in a JSON file instead of requiring Python code changes.

**Architecture:** Keep the built-in default mapping as a fallback, then merge mappings from a JSON file, then merge `OPENROUTER_MODEL_MAP_JSON` as the highest-priority override. The default file is `openrouter_model_map.json`, and `OPENROUTER_MODEL_MAP_FILE` can point to another file.

**Tech Stack:** Python 3.12, FastAPI, aiohttp, plain JSON configuration, existing script-style tests.

---

### Task 1: Add Focused Tests

**Files:**
- Create: `tests/openrouter/test_openrouter_model_map.py`
- Modify: none
- Test: `tests/openrouter/test_openrouter_model_map.py`

- [ ] **Step 1: Write tests for file loading, env override, and thinking aliases**

```python
#!/usr/bin/env python3

import importlib
import json


def reload_openrouter_proxy(monkeypatch, tmp_path, file_mapping=None, env_mapping=None):
    config_path = tmp_path / "openrouter_model_map.json"
    if file_mapping is not None:
        config_path.write_text(json.dumps(file_mapping), encoding="utf-8")

    monkeypatch.setenv("OPENROUTER_MODEL_MAP_FILE", str(config_path))
    if env_mapping is None:
        monkeypatch.delenv("OPENROUTER_MODEL_MAP_JSON", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_MODEL_MAP_JSON", json.dumps(env_mapping))

    import openrouter_proxy

    return importlib.reload(openrouter_proxy)


def test_load_model_map_from_json_file(monkeypatch, tmp_path):
    openrouter_proxy = reload_openrouter_proxy(
        monkeypatch,
        tmp_path,
        file_mapping={"claude-opus-4-7": "anthropic/claude-opus-4.7"},
    )

    model_map = openrouter_proxy.load_model_map_from_env()

    assert model_map["claude-opus-4-7"] == "anthropic/claude-opus-4.7"


def test_env_mapping_overrides_file_mapping(monkeypatch, tmp_path):
    openrouter_proxy = reload_openrouter_proxy(
        monkeypatch,
        tmp_path,
        file_mapping={"claude-opus-latest": "anthropic/claude-opus-4.7"},
        env_mapping={"claude-opus-latest": "anthropic/claude-opus-override"},
    )

    model_map = openrouter_proxy.load_model_map_from_env()

    assert model_map["claude-opus-latest"] == "anthropic/claude-opus-override"


def test_thinking_alias_uses_file_mapping(monkeypatch, tmp_path):
    openrouter_proxy = reload_openrouter_proxy(
        monkeypatch,
        tmp_path,
        file_mapping={"claude-opus-4-7": "anthropic/claude-opus-4.7"},
    )

    handler = openrouter_proxy.OpenAIRequestHandler("https://openrouter.ai/api", "test-key")
    request_data = {
        "model": "claude-opus-4-7-thinking",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    normalized = handler._normalize_model_and_reasoning(request_data)

    assert normalized["original_model"] == "claude-opus-4-7-thinking"
    assert normalized["model"] == "anthropic/claude-opus-4.7"
    assert normalized["thinking"] == {"type": "enabled", "budget_tokens": 30000}
    assert normalized["max_tokens"] == 80000
```

- [ ] **Step 2: Run tests and verify they fail before implementation**

Run: `uv run python tests/openrouter/test_openrouter_model_map.py`

Expected: at least `test_load_model_map_from_json_file` fails because `load_model_map_from_env()` does not read `OPENROUTER_MODEL_MAP_FILE` yet.

### Task 2: Implement JSON File Loading

**Files:**
- Modify: `openrouter_proxy.py:1-138`
- Test: `tests/openrouter/test_openrouter_model_map.py`

- [ ] **Step 1: Add a JSON mapping helper and merge file mapping before env mapping**

Update `openrouter_proxy.py` so `load_model_map_from_env()` uses this behavior:

```python
DEFAULT_OPENROUTER_MODEL_MAP_FILE = "openrouter_model_map.json"


def _normalize_model_map(raw_model_map: Any, source: str) -> Dict[str, str]:
    if not isinstance(raw_model_map, dict):
        logger.warning(f"{source} must be a JSON object, ignoring mapping")
        return {}

    return {
        str(alias).strip(): str(target).strip()
        for alias, target in raw_model_map.items()
        if str(alias).strip() and str(target).strip()
    }


def load_model_map_from_env() -> Dict[str, str]:
    model_map = DEFAULT_OPENROUTER_MODEL_MAP.copy()

    model_map_file = os.getenv("OPENROUTER_MODEL_MAP_FILE", DEFAULT_OPENROUTER_MODEL_MAP_FILE)
    if model_map_file and os.path.exists(model_map_file):
        try:
            with open(model_map_file, "r", encoding="utf-8") as file:
                file_model_map = json.load(file)
            model_map.update(_normalize_model_map(file_model_map, model_map_file))
        except json.JSONDecodeError as error:
            logger.warning(f"Failed to parse {model_map_file}: {error}")
        except OSError as error:
            logger.warning(f"Failed to read {model_map_file}: {error}")

    raw_model_map = os.getenv("OPENROUTER_MODEL_MAP_JSON")
    if raw_model_map:
        try:
            custom_model_map = json.loads(raw_model_map)
            model_map.update(_normalize_model_map(custom_model_map, "OPENROUTER_MODEL_MAP_JSON"))
        except json.JSONDecodeError as error:
            logger.warning(f"Failed to parse OPENROUTER_MODEL_MAP_JSON: {error}")

    return model_map
```

- [ ] **Step 2: Run focused tests**

Run: `uv run python tests/openrouter/test_openrouter_model_map.py`

Expected: all tests pass.

### Task 3: Add Default Config and Docs

**Files:**
- Create: `openrouter_model_map.json`
- Modify: `.env.example`
- Modify: `USAGE.md`

- [ ] **Step 1: Add default JSON file**

Create `openrouter_model_map.json`:

```json
{
  "claude-opus-4-7": "anthropic/claude-opus-4.7",
  "claude-opus-latest": "anthropic/claude-opus-4.7"
}
```

- [ ] **Step 2: Document the optional file path env var**

Add to `.env.example` near OpenRouter settings:

```env
# OpenRouter model alias mapping file
OPENROUTER_MODEL_MAP_FILE=openrouter_model_map.json
```

- [ ] **Step 3: Update usage docs**

Update `USAGE.md` to say OpenRouter aliases can be maintained in `openrouter_model_map.json`, with `OPENROUTER_MODEL_MAP_JSON` remaining an emergency override.

- [ ] **Step 4: Run syntax and focused tests**

Run: `uv run python -m py_compile openrouter_proxy.py anthropic_proxy.py`

Expected: no output and exit code 0.

Run: `uv run python tests/openrouter/test_openrouter_model_map.py`

Expected: all tests pass.
