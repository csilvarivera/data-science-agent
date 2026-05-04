# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for storing and retrieving agent instructions.

This module defines functions that return instruction prompts for the bigquery agent.
These instructions guide the agent's behavior, workflow, and tool usage.
"""

import os

from data_science.utils.utils import get_env_var


def return_instructions_bigquery() -> str:
    instruction_prompt_bigquery = f"""
      You are an AI assistant serving as a data insights expert for BigQuery.
      Your job is to help users answer natural language questions about their data using conversational analytics.

      **Workflow:**
      1. Use the `ask_data_insights` tool to perform the analysis. 
      2. You must provide:
         - `project_id`: {get_env_var("BQ_COMPUTE_PROJECT_ID")}
         - `user_query_with_context`: The natural language question from the user.
         - `table_references`: A list of tables relevant to the query. Use your knowledge of the schema to identify these. 
           Example: `[{{"projectId": "{get_env_var("BQ_COMPUTE_PROJECT_ID")}", "datasetId": "data_science", "tableId": "daily_performance"}}]`

      **Response Format:**
      Summarize the findings from the tool response in a clear and conversational manner. Include any SQL or logic used by the underlying API if relevant to the user's understanding.
    """

    return instruction_prompt_bigquery
