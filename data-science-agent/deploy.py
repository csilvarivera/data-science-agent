"""Deployment script for MRNA."""

import os
import sys

# Now use an absolute import from the project root
import vertexai
from vertexai import agent_engines
from vertexai.preview import reasoning_engines
from dotenv import load_dotenv
from data_science.agent import root_agent


# load the environment
load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION")
GOOGLE_CLOUD_STORAGE_BUCKET = os.getenv("MODEL_STAGING_BUCKET")
BQ_COMPUTE_PROJECT_ID = os.getenv("BQ_COMPUTE_PROJECT_ID")
BQ_DATA_PROJECT_ID = os.getenv("BQ_DATA_PROJECT_ID")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID")
DATASET_CONFIG_FILE = os.getenv("DATASET_CONFIG_FILE")


print("Project ID:", PROJECT_ID)
print("Location:", LOCATION)
print("Staging bucket:", GOOGLE_CLOUD_STORAGE_BUCKET)
if not PROJECT_ID or not LOCATION or not GOOGLE_CLOUD_STORAGE_BUCKET:
  print(
      "Missing GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, or STAGING_BUCKET",
      file=sys.stderr,
  )
  sys.exit(1)

vertexai.init(
    project=PROJECT_ID,
    location=VERTEX_LOCATION,
    staging_bucket=GOOGLE_CLOUD_STORAGE_BUCKET,
)

def test_local_agent():
  
  # create a local version of your root agent
  app = reasoning_engines.AdkApp(
    agent=root_agent,
    enable_tracing=True,
  )

  
  session = app.create_session(user_id="u_123")
  session
  for event in app.stream_query(
    user_id="u_123",
    session_id=session.id,
    message="whats the weather in new york",
  ):
    print(event)


def deploy_agent():
  # Deploy to AgentEngine - Check Cloud Logging for detailed issues.
  remote_agent = agent_engines.create(
      root_agent,
          requirements=[
              "python-dotenv>=1.0.1",
              "google-adk>=1.22.0",
              "immutabledict>=4.2.1",
              "sqlglot>=26.10.1",
              "db-dtypes>=1.4.2",
              "regex>=2024.11.6",
              "tabulate>=0.9.0",
              "google-cloud-aiplatform[adk,agent-engines]>=1.93.0",
              "absl-py>=2.2.2",
              "pydantic>=2.11.3",
              "pandas>=2.3.0",
              "numpy>=2.3.1",
              "toolbox-core>=0.3.0",
              "opentelemetry-sdk>=1.36.0",
              "opentelemetry-exporter-otlp-proto-http>=1.36.0",
              "pg8000>=1.31.2",
          ],
      extra_packages= ["data_science/agent.py",
                "data_science/tools.py",
                "data_science/prompts.py",
                "data_science/sub_agents/",
                "data_science/utils"],
      display_name="data_science_agent",
      env_vars = {
        "GOOGLE_CLOUD_LOCATION": "global",
        "BQ_COMPUTE_PROJECT_ID": BQ_COMPUTE_PROJECT_ID,
        "BQ_DATA_PROJECT_ID": BQ_DATA_PROJECT_ID,
        "BQ_DATASET_ID": BQ_DATASET_ID,
        "DATASET_CONFIG_FILE": DATASET_CONFIG_FILE,
        "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY": "true",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
      }
  )
  print(f"\nSuccessfully created agent: {remote_agent.resource_name}")


def list_agents():
  for agent in agent_engines.list():
    print(agent.display_name)

def call_agent():
  # Change to your agent engine ID
  AGENT_ENGINE_ID="projects/774298971519/locations/us-central1/reasoningEngines/5532474230131654656"
  agent = agent_engines.get(AGENT_ENGINE_ID)

  remote_session = agent.create_session(user_id="u_123")
  print (f" calling agent {AGENT_ENGINE_ID} with session {remote_session['id']}")
  for event in agent.stream_query(
    user_id="u_123",
    session_id=remote_session['id'],
    # session_id="5286711940846977024",
    message="Hello how can you help",
  ):
    print(event)


if __name__ == "__main__":
    try:
        
        # Call the deployment function with the obtained values
        deploy_agent()
        print("\nDeployment script finished.")
        #call_agent()
        
    except (ValueError, FileNotFoundError) as e: # Catch specific known errors
         print(f"Configuration Error: {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e: # Catch any other unexpected errors during the process
        print(f"Script execution failed: {e}", file=sys.stderr)
        sys.exit(1)

