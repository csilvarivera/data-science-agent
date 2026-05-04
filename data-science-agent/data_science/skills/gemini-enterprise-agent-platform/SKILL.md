---
name: gemini-enterprise-agent-platform
description: >
  Core constraints and guidelines for authoring pure scikit-learn or custom machine learning training runs
  on data volumes up to 100MB inside the sandboxed runtime container.
metadata:
  author: Google-Partner
  license: Apache-2.0
---

# Gemini Enterprise Agent Platform Modeling Guide

When performing custom operations on input datasets:

1. **Retrieve Platform Design Logic (L3 Guide):**
   - Activate and review `references/platform_guide.md` using the `load_skill_resource` tool to gather concrete code architecture parameters.

2. **Extraction & Staging (Passing up to 100MB):**
   - For efficiency reasons, large files (e.g., 100MB) cannot transfer over standard request payloads.
   - Staging strategy involves executing BigQuery export actions directly to target environment staging locations (`gs://...`).
   - Target location tool: `export_bq_to_gcs`.

3. **Sandbox Sandbox Environment Execution:**
   - Trigger isolated code sandboxing operations via `execute_script_sandbox_interactive`.
   - The function requires a Python source file block text assigned inside parameter string key `code`.
   - Container allocations execute dynamically without requiring individual session instantiation operations.
   - Only use Python traditional machine learning suites (`scikit-learn`, `tensorflow`). Avoid invoking BQML wrapper commands.

4. **Model Artifact Persistence Registry:**
   - Serialize results in sandbox to binary object via `joblib`.
   - Push output to system model registers using the platform `aiplatform` integration logic.
