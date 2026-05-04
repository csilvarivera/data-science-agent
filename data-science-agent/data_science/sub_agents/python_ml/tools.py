"""Tools for Python ML sub-agent."""

import os
import logging
from google.cloud import bigquery
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)


def export_bq_to_gcs(table_id: str) -> dict:
    """Export a BigQuery table directly to an internal Google Cloud Storage bucket.

    Args:
        table_id: Fully qualified path to target BQ table formatted as "project.dataset.table".

    Returns:
        Dict outcome containing the "gcs_uri" location string.
    """
    logger.info(f"Extracting data from table: {table_id}")
    try:
        bq_location = os.getenv("BQ_LOCATION", "US")
        client = bigquery.Client(location=bq_location)
        
        bucket_uri = os.getenv("MODEL_STAGING_BUCKET", "")
        if not bucket_uri:
            raise ValueError("Missing definition for MODEL_STAGING_BUCKET within runtime variables.")

        # Strip prefix
        bucket = bucket_uri.replace("gs://", "").rstrip("/")
        
        table_id_clean = table_id.replace(":", ".")
        parts = table_id_clean.split(".")
        if len(parts) < 3:
            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise ValueError("Missing GOOGLE_CLOUD_PROJECT within runtime variables.")
            if len(parts) == 2:
                parts = [project] + parts
            else:
                raise ValueError(f"Invalid table identifier: {table_id}")

        project, dataset_name, table_name = parts[0], parts[1], parts[2]
        destination_uri = f"gs://{bucket}/dataset_exports/{table_name}.csv"
        
        dataset_ref = client.dataset(dataset_name, project=project)
        table_ref = dataset_ref.table(table_name)

        extract_job = client.extract_table(
            table_ref,
            destination_uri,
            location=bq_location
        )
        extract_job.result()
        logger.info(f"Finished exporting data into URI: {destination_uri}")
        
        return {
            "status": "success",
            "gcs_uri": destination_uri,
            "table_id": table_id
        }
        
    except Exception as e:
        logger.error(f"Error running BigQuery extract operation: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def execute_sandbox_ml_script(code: str, tool_context: ToolContext) -> dict:
    """Executes the generated Python machine learning script inside the isolated sandbox execution tool.

    Args:
        code: Program string containing scikit-learn algorithm configurations.

    Returns:
        Outcome summary with stdout or status information.
    """
    logger.info("Executing Python training sandbox sequence via confirmation interface")
    from google.cloud import storage, aiplatform
    from google.adk.code_executors.code_execution_utils import CodeExecutionInput, File
    
    from .agent import get_code_executor
    
    try:
        # 1. Download staged CSV data locally from GCS on the host machine
        logger.info("Initializing local Google Cloud Storage client...")
        storage_client = storage.Client()
        
        bucket_uri = os.getenv("MODEL_STAGING_BUCKET", "")
        if not bucket_uri:
            raise ValueError("Missing definition for MODEL_STAGING_BUCKET within runtime variables.")
            
        bucket_name = bucket_uri.replace("gs://", "").rstrip("/")
        bucket_obj = storage_client.bucket(bucket_name)
        
        logger.info(f"Locating staged CSV dataset in bucket: gs://{bucket_name}/dataset_exports/...")
        blobs = bucket_obj.list_blobs(prefix="dataset_exports/")
        csv_blob = None
        for b in blobs:
            logger.info(f"  Found staged blob: gs://{bucket_name}/{b.name} (size: {b.size} bytes)")
            if b.name.endswith(".csv"):
                csv_blob = b
                
        if not csv_blob:
            raise FileNotFoundError(f"No staged dataset CSV file found inside storage bucket path: gs://{bucket_name}/dataset_exports/")
            
        logger.info(f"Downloading staged dataset from GCS: gs://{bucket_name}/{csv_blob.name} (size: {csv_blob.size} bytes)...")
        csv_bytes = csv_blob.download_as_bytes()
        logger.info("Staged CSV dataset successfully downloaded as bytes from GCS.")
        
        # 2. Package dataset CSV as an input file to the Sandbox container
        csv_text = csv_bytes.decode("utf-8", errors="ignore")
        
        # Truncate dataset to first 1000 rows to prevent exceeding HTTP JSON payload limits
        lines = csv_text.splitlines()
        truncated_text = "\n".join(lines[:1000])
        logger.info(f"Truncated staged CSV to first 1000 rows for safe HTTP JSON transmission (size: {len(truncated_text)} characters)")
        
        input_file = File(name="input.csv", content=truncated_text, mime_type="text/csv")
        
        # Check if we can reuse a cached sandbox from previous run
        from google.adk.code_executors.agent_engine_sandbox_code_executor import AgentEngineSandboxCodeExecutor
        
        sandbox_name = tool_context.state.get("sandbox_resource_name")
        if sandbox_name:
            logger.info(f"Reusing active Sandbox container: {sandbox_name}")
            executor = AgentEngineSandboxCodeExecutor(sandbox_resource_name=sandbox_name)
        else:
            logger.info("Obtaining fresh sandboxed code executor instance...")
            executor = get_code_executor()
            
        invocation_context = tool_context._invocation_context
        
        logger.info("Dispatching Python script to Cloud Sandbox container environment...")
        logger.debug(f"Generated script code length: {len(code)} characters")
        
        result = executor.execute_code(
            invocation_context,
            CodeExecutionInput(code=code, input_files=[input_file])
        )
        
        # Cache the created sandbox resource name for future reuse
        if not sandbox_name and executor.sandbox_resource_name:
            tool_context.state["sandbox_resource_name"] = executor.sandbox_resource_name
            logger.info(f"Cached active Sandbox resource name for reuse: {executor.sandbox_resource_name}")
        
        logger.info("Sandbox code execution completed successfully.")
        logger.debug(f"Execution stdout output: {result.stdout}")
        logger.debug(f"Execution stderr output: {result.stderr}")
        
        if not result.stdout and not result.stderr:
            logger.warning("Warning: Sandbox execution returned completely empty stdout and stderr.")
            
        # 3. Check if trained model binary was returned from Sandbox container
        logger.info(f"Inspecting sandbox execution return files. Total returned files: {len(result.output_files)}")
        model_bytes = None
        for f in result.output_files:
            logger.info(f"  Captured output file: {f.name} (content length: {len(f.content) if f.content else 0} bytes)")
            if f.name == "model.joblib":
                # If content is string (base64), decode it to bytes
                if isinstance(f.content, str):
                    import base64
                    model_bytes = base64.b64decode(f.content)
                else:
                    model_bytes = f.content
                break
                
        if model_bytes:
            logger.info("Trained model binary successfully captured from Sandbox container output files.")
            
            # Upload model binary bytes to GCS
            gcs_model_path = "models/model.joblib"
            dest_blob = bucket_obj.blob(gcs_model_path)
            dest_blob.upload_from_string(model_bytes, content_type="application/octet-stream")
            gcs_model_uri = f"gs://{bucket_name}/{gcs_model_path}"
            logger.info(f"Trained model successfully staged in GCS: {gcs_model_uri}")
            
            # Register model directly in Vertex AI Model Registry
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            logger.info(f"Registering model dynamically in Vertex AI Model Registry (project: {project_id})...")
            aiplatform.init(project=project_id, location="us-central1")
            model_resource = aiplatform.Model.upload(
                display_name="churn_prediction_model",
                artifact_uri=f"gs://{bucket_name}/models/",
                serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
            )
            logger.info(f"Model successfully registered: {model_resource.resource_name}")
            
        return {
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "status": "success" if not result.stderr else "failed",
            "error_details": None
        }
    except Exception as e:
        logger.error(f"Fatal error encountered during Sandbox execution: {e}", exc_info=True)
        return {
            "stdout": "",
            "stderr": "",
            "status": "error",
            "error_details": str(e)
        }
