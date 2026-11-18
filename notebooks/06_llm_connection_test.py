# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 06 - LLM connection test (Databricks endpoint OR free external API)
# MAGIC Run the cells in order. Needs `04_agent_tools` in the same folder.
# MAGIC
# MAGIC 1. Set `PROVIDER` in the config cell: `databricks`, `groq` or `gemini`.
# MAGIC 2. For `groq` / `gemini`, paste your API key in the widget at the top of the notebook (never in a cell).
# MAGIC 3. The last cell prints `LLM CONNECTION OK` when the tool-calling loop works.

# COMMAND ----------

# MAGIC %pip install openai databricks-sdk --upgrade --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./04_agent_tools

# COMMAND ----------

# Config
PROVIDER = "databricks"        # "databricks" | "groq" | "gemini"
MODEL = ""                     # leave empty to auto-pick (databricks) or use the default below

EXTERNAL = {
    "groq":   {"base_url": "https://api.groq.com/openai/v1",
               "default_model": "llama-3.3-70b-versatile"},      # check the provider's current model names
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "default_model": "gemini-2.5-flash"},             # check the provider's current model names
}

dbutils.widgets.text("api_key", "", "API key (groq / gemini only)")

# COMMAND ----------

# Reachability test (a 401/403 is fine: it means the domain is reachable and only the key is missing)
import requests

for name, url in {
    "Groq": "https://api.groq.com/openai/v1/models",
    "Gemini": "https://generativelanguage.googleapis.com/v1beta/openai/models",
    "OpenRouter": "https://openrouter.ai/api/v1/models",
}.items():
    try:
        r = requests.get(url, timeout=8)
        print(f"{name:10s} -> reachable, HTTP {r.status_code}")
    except Exception as e:
        print(f"{name:10s} -> blocked: {type(e).__name__}")

# COMMAND ----------

# List Databricks serving endpoints (only informative; skip if you use an external provider)
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
try:
    for e in w.serving_endpoints.list():
        state = e.state.ready.value if (e.state and e.state.ready) else "?"
        print(e.name, "|", getattr(e, "task", None), "|", state)
except Exception as e:
    print("Could not list endpoints:", type(e).__name__, str(e)[:200])

# COMMAND ----------

# Build the client and pick the model
from openai import OpenAI

def _quick_ok(client, model):
    try:
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Réponds uniquement par le mot OK."}],
            max_tokens=10, temperature=0)
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"ERROR {type(e).__name__}: {str(e)[:160]}"

if PROVIDER == "databricks":
    client = w.serving_endpoints.get_open_ai_client()
    if MODEL:
        candidates = [MODEL]
    else:
        names = [e.name for e in w.serving_endpoints.list()]
        candidates = [n for n in names
                      if n.startswith("databricks-")
                      and not any(x in n for x in ("gte", "bge", "embed"))]
        print("Candidate chat endpoints:", candidates or "none found")
    MODEL = None
    for cand in candidates:
        out = _quick_ok(client, cand)
        print(f"  {cand} -> {out}")
        if not out.startswith("ERROR"):
            MODEL = cand
            break
    if MODEL is None:
        raise RuntimeError("No usable Databricks chat endpoint. Switch PROVIDER to 'groq' or 'gemini'.")
else:
    api_key = dbutils.widgets.get("api_key").strip()
    if not api_key:
        raise RuntimeError("Paste your API key in the 'api_key' widget at the top of the notebook, then rerun this cell.")
    cfg = EXTERNAL[PROVIDER]
    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    MODEL = MODEL or cfg["default_model"]
    out = _quick_ok(client, MODEL)
    print(f"{MODEL} -> {out}")
    if out.startswith("ERROR"):
        raise RuntimeError("Simple call failed. Check the key, the model name and the reachability test above.")

print(f"\nProvider: {PROVIDER} | Model: {MODEL}")

# COMMAND ----------

# Tool-calling loop (reused later by the agent)
import json

def run_tool_loop(question, model, tools, system=None, max_steps=6, verbose=True):
    """The LLM asks for tools, we run them, we send the results back, until a final answer."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    tool_names_called = []

    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools,
            tool_choice="auto", temperature=0, max_tokens=1000)
        msg = resp.choices[0].message

        if not msg.tool_calls:                      # final answer
            return msg.content, messages, tool_names_called

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls]})

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if verbose:
                print(f"[step {step}] tool: {tc.function.name}({args})")
            tool_names_called.append(tc.function.name)
            result = call_tool(tc.function.name, args)
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

    return "STOP: max_steps reached without a final answer", messages, tool_names_called

# COMMAND ----------

import re

answer, history, called = run_tool_loop(
    "Pour CASE_004, donne les prévisions Batch et Real-Time et l'écart. "
    "Utilise l'outil get_forecasts.",
    MODEL, TOOL_SCHEMAS)

print("\n--- FINAL ANSWER ---")
print(answer)

def _digits_only(s):
    # "1 200", "1 200" (non-breaking space), "1,200" -> "1200"
    return re.sub(r"[\s\u00a0\u202f,.]", "", s or "")

flat = _digits_only(answer)
assert "get_forecasts" in called, "The model did not call get_forecasts. Try another model."
assert all(v in flat for v in ("900", "1200", "300")), "The answer does not contain 900, 1200 and 300."
print("\nLLM CONNECTION OK")