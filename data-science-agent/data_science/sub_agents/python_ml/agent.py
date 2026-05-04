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

"""Python Machine Learning sub-agent."""

import logging
import os

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool
from google.adk.code_executors.agent_engine_sandbox_code_executor import AgentEngineSandboxCodeExecutor

from data_science.sub_agents.bigquery.agent import bigquery_agent
from data_science.sub_agents.bigquery.tools import (
    get_database_settings as get_bq_database_settings,
)
from data_science.sub_agents.bqml.tools import rag_response
from data_science.sub_agents.python_ml.prompts import return_instructions_python_ml
from data_science.sub_agents.python_ml.tools import export_bq_to_gcs, execute_sandbox_ml_script

logger = logging.getLogger(__name__)


def setup_before_agent_call(callback_context: CallbackContext):
    """Setup configuration in the context prior to running model execution tasks."""

    if "database_settings" in callback_context.state:
        return

    db_settings = {
        "bigquery": get_bq_database_settings(),
    }
    callback_context.state["database_settings"] = db_settings

    schema = callback_context.state["database_settings"]["bigquery"]["schema"]

    callback_context._invocation_context.agent.instruction = (
        return_instructions_python_ml()
        + f"""

    <The BigQuery schema of the relevant data with a few sample rows>
    {schema}
    </The BigQuery schema of the relevant data with a few sample rows>
    """
    )


async def call_db_agent(
    question: str,
    tool_context: ToolContext,
):
    """Tool to call database (nl2sql) agent."""

    agent_tool = AgentTool(agent=bigquery_agent)
    db_agent_output = await agent_tool.run_async(
        args={"request": question}, tool_context=tool_context
    )
    tool_context.state["db_agent_output"] = db_agent_output
    return db_agent_output


def get_code_executor():
    """Returns the mandatory AgentEngineSandboxCodeExecutor implementation."""
    logger.info("Initializing Python code interpreter sandbox via AgentEngineSandboxCodeExecutor")
    
    # Temporarily override GOOGLE_CLOUD_LOCATION to ensure sandbox runs in a supported region
    orig_loc = os.environ.get("GOOGLE_CLOUD_LOCATION")
    os.environ["GOOGLE_CLOUD_LOCATION"] = os.getenv("VERTEX_LOCATION", "us-central1")
    try:
        executor = AgentEngineSandboxCodeExecutor()
    finally:
        if orig_loc is not None:
            os.environ["GOOGLE_CLOUD_LOCATION"] = orig_loc
        else:
            del os.environ["GOOGLE_CLOUD_LOCATION"]
            
    return executor


import pathlib
from google.adk.skills import load_skill_from_dir
from google.adk.tools import FunctionTool, ToolContext
from google.adk.tools.skill_toolset import SkillToolset

execute_script_sandbox_interactive = FunctionTool(
    execute_sandbox_ml_script,
    require_confirmation=True
)

# Initialize direct SkillToolset module properties instance
skills_path = pathlib.Path(__file__).parent.parent.parent / "skills" / "gemini-enterprise-agent-platform"
platform_skill = load_skill_from_dir(skills_path)
skill_toolset = SkillToolset(skills=[platform_skill])


python_ml_agent = Agent(
    model="gemini-3-flash-preview",
    name="python_ml_agent",
    instruction=return_instructions_python_ml(),
    before_agent_callback=setup_before_agent_call,
    tools=[call_db_agent, export_bq_to_gcs, execute_script_sandbox_interactive, rag_response, skill_toolset],
    code_executor=None,
)
