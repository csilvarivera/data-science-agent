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

"""Tools for the ADK Sampmles Data Science Agent."""

import logging

from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool

from .sub_agents import bigquery_agent, bqml_agent

logger = logging.getLogger(__name__)


async def call_bqml_agent(
    question: str,
    tool_context: ToolContext,
):
    """Tool to call BigQuery ML agent for model training and deployment.
    
    Args:
        question (str): Natural language question or task for BQML.
        tool_context (ToolContext): The tool context.
        
    Returns:
        Response from the BQML agent.
    """
    logger.debug("call_bqml_agent: %s", question)

    agent_tool = AgentTool(agent=bqml_agent)

    bqml_agent_output = await agent_tool.run_async(
        args={"request": question}, tool_context=tool_context
    )
    tool_context.state["bqml_agent_output"] = bqml_agent_output
    return bqml_agent_output


async def call_bigquery_agent(
    question: str,
    tool_context: ToolContext,
):
    """Tool to call bigquery database (nl2sql) agent."""
    logger.debug("call_bigquery_agent: %s", question)

    agent_tool = AgentTool(agent=bigquery_agent)

    bigquery_agent_output = await agent_tool.run_async(
        args={"request": question}, tool_context=tool_context
    )
    tool_context.state["bigquery_agent_output"] = bigquery_agent_output
    return bigquery_agent_output


async def call_python_ml_agent(
    question: str,
    tool_context: ToolContext,
):
    """Tool to call Python machine learning sub-agent.
    
    Args:
        question (str): Natural language request or parameter string.
        tool_context (ToolContext): The execution context.
        
    Returns:
        Response text indicating code outcome summary.
    """
    logger.debug("call_python_ml_agent: %s", question)
    from .sub_agents.python_ml.agent import python_ml_agent

    agent_tool = AgentTool(agent=python_ml_agent)

    python_agent_output = await agent_tool.run_async(
        args={"request": question}, tool_context=tool_context
    )
    tool_context.state["python_ml_agent_output"] = python_agent_output
    return python_agent_output

