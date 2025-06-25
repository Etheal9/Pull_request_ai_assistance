

import os
import json
import logging
import traceback # Import traceback for detailed error logging

from fastmcp import FastMCP
from huggingface_hub import HfApi, model_info, ModelCard, ModelCardData
from huggingface_hub.utils import HfHubHTTPError
from dotenv import load_dotenv # Used for local development to load .env file

import gradio as gr
# Configure basic logging to see messages in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file for local development.
# Note: When running with `uv run` and pyproject.toml, uv might handle dotenv loading,
# but keeping this ensures explicit loading if not.
load_dotenv()

# --- Configuration and FastMCP Server Initialization ---
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize HF API client
# The client is created conditionally to allow the server to start even if the token is missing,
# but tools will report errors if the token is needed and not available.
hf_api = HfApi(token=HF_TOKEN) if HF_TOKEN else None

# Create the FastMCP server instance with a descriptive name
mcp = FastMCP("hf-tagging-bot")

# --- MCP Tool: get_current_tags ---
@mcp.tool()
def get_current_tags(repo_id: str) -> str:
    """
    Get current tags from a Hugging Face model repository.

    Args:
        repo_id: The full repository ID (e.g., "openai/gpt-2", "username/my-model").

    Returns:
        A JSON string containing the status, repo_id, current_tags (list of strings),
        and count of tags. Returns an error message if the token is not configured
        or an exception occurs during API call.
    """
    logging.info(f"🔧 get_current_tags called with repo_id: {repo_id}")

    if not hf_api:
        error_result = {"error": "HF token not configured. Cannot fetch tags without authentication."}
        json_str = json.dumps(error_result)
        logging.error(f"❌ No HF API token - returning: {json_str}")
        return json_str

    try:
        logging.info(f"📡 Fetching model info for: {repo_id}")
        info = model_info(repo_id=repo_id, token=HF_TOKEN)
        # Extract tag names from the info object's tags attribute
        current_tags = info.tags if info.tags else []
        logging.info(f"🏷️ Found {len(current_tags)} tags for {repo_id}: {current_tags}")

        result = {
            "status": "success",
            "repo_id": repo_id,
            "current_tags": current_tags,
            "count": len(current_tags),
        }
        json_str = json.dumps(result)
        logging.info(f"✅ get_current_tags returning: {json_str}")
        return json_str

    except HfHubHTTPError as e:
        # Handle specific Hugging Face Hub HTTP errors (e.g., repo not found, permission denied)
        logging.error(f"❌ HF Hub HTTP Error in get_current_tags for {repo_id}: {e}")
        error_result = {"status": "error", "repo_id": repo_id, "error": str(e), "message": f"Hugging Face Hub API error: {e.response.status_code} - {e.response.reason}"}
        json_str = json.dumps(error_result)
        logging.error(f"❌ get_current_tags error returning: {json_str}")
        return json_str
    except Exception as e:
        # Catch any other unexpected exceptions
        logging.error(f"❌ Unexpected Error in get_current_tags for {repo_id}: {str(e)}")
        logging.error(f"❌ Traceback: {traceback.format_exc()}") # Print full traceback for debugging
        error_result = {"status": "error", "repo_id": repo_id, "error": str(e), "message": "An unexpected error occurred while retrieving tags."}
        json_str = json.dumps(error_result)
        logging.error(f"❌ get_current_tags error returning: {json_str}")
        return json_str

# --- MCP Tool: add_new_tag ---
@mcp.tool()
def add_new_tag(repo_id: str, new_tag: str) -> str:
    """
    Add a new tag to a Hugging Face model repository by creating a pull request.

    This function fetches the current model card, updates its tags, and then
    creates a commit to update the README.md via a pull request on the Hub.

    Args:
        repo_id: The full repository ID (e.g., "openai/gpt-2").
        new_tag: The new tag string to add.

    Returns:
        A JSON string indicating the status of the operation (success, already_exists, error),
        the repository ID, the tag, and potentially the PR URL.
    """
    logging.info(f"🔧 add_new_tag called with repo_id: {repo_id}, new_tag: {new_tag}")

    if not hf_api:
        error_result = {"error": "HF token not configured. Cannot add tags without authentication."}
        json_str = json.dumps(error_result)
        logging.error(f"❌ No HF API token - returning: {json_str}")
        return json_str

    try:
        # Get current model info and tags to check for existing tag and get current state
        logging.info(f"📡 Fetching current model info for: {repo_id}")
        info = model_info(repo_id=repo_id, token=HF_TOKEN)
        current_tags = info.tags if info.tags else []
        logging.info(f"🏷️ Current tags for {repo_id}: {current_tags}")

        # Validate before acting: Check if the tag already exists
        if new_tag in current_tags:
            logging.warning(f"⚠️ Tag '{new_tag}' already exists in {repo_id}'s tags: {current_tags}")
            result = {
                "status": "already_exists",
                "repo_id": repo_id,
                "tag": new_tag,
                "message": f"Tag '{new_tag}' already exists in repository '{repo_id}'",
                "current_tags": current_tags,
            }
            json_str = json.dumps(result)
            logging.info(f"🏷️ add_new_tag (already exists) returning: {json_str}")
            return json_str

        # Prepare the updated tag list
        updated_tags = current_tags + [new_tag]
        logging.info(f"🆕 Will update tags from {current_tags} to {updated_tags}")

        # --- Model Card Management ---
        # The Hugging Face Hub stores metadata (including tags) in the README.md (Model Card).
        # We need to load/create it to update tags.
        try:
            # Load existing model card
            logging.info(f"📄 Loading existing model card for {repo_id}...")
            card = ModelCard.load(repo_id, token=HF_TOKEN)
            # Ensure the ModelCard has a 'data' attribute for metadata.
            if not hasattr(card, "data") or card.data is None:
                card.data = ModelCardData()
            logging.info("📄 Existing model card loaded.")
        except HfHubHTTPError as e:
            # If ModelCard.load fails (e.g., 404 for no README.md), create a new one.
            if e.response.status_code == 404:
                logging.warning(f"📄 No existing model card found for {repo_id}. Creating a new one.")
                card = ModelCard("") # Start with empty content
                card.data = ModelCardData() # Initialize ModelCardData
            else:
                raise # Re-raise other HTTP errors

        # Update tags in the ModelCardData object
        # Convert to dict, update tags, then convert back to ModelCardData to ensure validation
        card_dict = card.data.to_dict()
        card_dict["tags"] = updated_tags
        card.data = ModelCardData(**card_dict)
        logging.info("📄 Model card data updated with new tags.")

        # --- Pull Request Creation ---
        pr_title = f"Add tag: '{new_tag}' to {repo_id}"
        pr_description = f"""
## Add tag: `{new_tag}`

This PR adds the `{new_tag}` tag to the model repository's `README.md` (Model Card).

**Changes:**
- Added `{new_tag}` to the 'tags' metadata in the model card.
- Updated from {len(current_tags)} to {len(updated_tags)} tags in total.

**Current tags:** {", ".join(current_tags) if current_tags else "None"}
**New tags (after merge):** {", ".join(updated_tags)}

---
🤖 This is a pull request created automatically by the Hugging Face Hub Tagging Bot.
"""
        logging.info(f"🚀 Creating PR with title: '{pr_title}' for {repo_id}")

        # The CommitOperationAdd operation is used to specify how files are changed in a commit.
        # Here, we're replacing the README.md content with the updated model card.
        # `create_pr=True` is key to automatically generating a pull request.
        from huggingface_hub import CommitOperationAdd # Re-import just in case, though it's at the top.

        commit_info = hf_api.create_commit(
            repo_id=repo_id,
            operations=[
                CommitOperationAdd(
                    path_in_repo="README.md",
                    # Encode the ModelCard object (which is markdown text) to bytes
                    path_or_fileobj=str(card).encode("utf-8")
                )
            ],
            commit_message=pr_title,
            commit_description=pr_description,
            token=HF_TOKEN,
            create_pr=True, # This flag makes it a PR instead of a direct commit to main
        )

        # The `create_commit` method returns different objects depending on whether a PR was created.
        # We try to get the PR URL if available, otherwise just use the commit info string.
        pr_url = getattr(commit_info, "pr_url", str(commit_info))

        logging.info(f"✅ PR created successfully! URL: {pr_url}")

        result = {
            "status": "success",
            "repo_id": repo_id,
            "tag": new_tag,
            "pr_url": pr_url,
            "previous_tags": current_tags,
            "new_tags": updated_tags,
            "message": f"Created PR to add tag '{new_tag}' to '{repo_id}'",
        }
        json_str = json.dumps(result)
        logging.info(f"✅ add_new_tag success returning: {json_str}")
        return json_str

    except HfHubHTTPError as e:
        logging.error(f"❌ HF Hub HTTP Error in add_new_tag for {repo_id}: {e}")
        error_result = {"status": "error", "repo_id": repo_id, "tag": new_tag, "error": str(e), "message": f"Hugging Face Hub API error: {e.response.status_code} - {e.response.reason}. Check token permissions or repository ID."}
        json_str = json.dumps(error_result)
        logging.error(f"❌ add_new_tag error returning: {json_str}")
        return json_str
    except Exception as e:
        # Catch any other unexpected exceptions and provide full traceback
        logging.error(f"❌ Unexpected Error in add_new_tag for {repo_id}: {str(e)}")
        logging.error(f"❌ Traceback: {traceback.format_exc()}")
        error_result = {
            "status": "error",
            "repo_id": repo_id,
            "tag": new_tag,
            "error": str(e),
            "message": "An unexpected error occurred while adding the tag and creating the PR."
        }
        json_str = json.dumps(error_result)
        logging.error(f"❌ add_new_tag error returning: {json_str}")
        return json_str

# The FastMCP server instance (`mcp`) is now ready to serve these tools.
# The `app.py` (your webhook listener) will import this `mcp` instance
# and use `mcp.app` to serve the tools.



# --- Gradio Interface for MCP Server Tools ---

# Wrapper for get_current_tags for Gradio UI
def gradio_get_current_tags(repo_id: str) -> str:
    """Gradio wrapper for get_current_tags MCP tool."""
    # Call the actual MCP tool function and return its JSON string output
    result = get_current_tags(repo_id)
    # The output is already JSON, so we can return it as a string
    return result

# Wrapper for add_new_tag for Gradio UI
def gradio_add_new_tag(repo_id: str, new_tag: str) -> str:
    """Gradio wrapper for add_new_tag MCP tool."""
    # Define a simple commit message for manual Gradio testing
    commit_message = f"Add tag '{new_tag}' via Gradio interface"
    result = add_new_tag(repo_id, new_tag, commit_message)
    return result

# You could also add a wrapper for remove_tags_from_model here if desired.
# def gradio_remove_tags_from_model(repo_id: str, tags_to_remove_str: str) -> str:
#     tags_to_remove = [tag.strip() for tag in tags_to_remove_str.split(',') if tag.strip()]
#     commit_message = f"Remove tags {', '.join(tags_to_remove)} via Gradio interface"
#     result = remove_tags_from_model(repo_id, tags_to_remove, commit_message)
#     return result

# Create the Gradio interface
# We define multiple functions (tabs) within a gr.Blocks context for a multi-page interface.
with gr.Blocks(title="HF Tagging MCP Tools Demo") as demo:
    gr.Markdown("# Hugging Face Tagging MCP Tools Demo")
    gr.Markdown("Directly test `get_current_tags` and `add_new_tag` tools.")

    with gr.Tab("Get Current Tags"):
        gr.Markdown("### Retrieve all tags for a Hugging Face repository.")
        repo_id_get = gr.Textbox(label="Repository ID (e.g., 'username/model-name')", placeholder="HuggingFaceH4/zephyr-7b-alpha")
        get_tags_button = gr.Button("Get Tags")
        current_tags_output = gr.JSON(label="Current Tags (JSON Output)") # Use gr.JSON for structured output
        get_tags_button.click(
            fn=gradio_get_current_tags,
            inputs=[repo_id_get],
            outputs=current_tags_output
        )

    with gr.Tab("Add New Tag"):
        gr.Markdown("### Add a new tag to a Hugging Face repository via Pull Request.")
        repo_id_add = gr.Textbox(label="Repository ID", placeholder="your_username/your_test_model")
        new_tag_input = gr.Textbox(label="New Tag to Add", placeholder="my-awesome-tag")
        add_tag_button = gr.Button("Add Tag via PR")
        add_tag_output = gr.JSON(label="Tag Addition Result (JSON Output)") # Use gr.JSON for structured output
        add_tag_button.click(
            fn=gradio_add_new_tag,
            inputs=[repo_id_add, new_tag_input],
            outputs=add_tag_output
        )

# Launch the interface and MCP server
if __name__ == "__main__":
    # Ensure the HF_TOKEN is actually loaded before launching.
    if not HF_TOKEN:
        logging.error("HF_TOKEN is not set. Gradio app might not function correctly. Please check your .env file.")
        
    # The `mcp_server=True` argument for demo.launch() means Gradio will run its UI
    # and also serve the MCP tools defined in this very script.
    # This creates a self-contained demo.
    demo.launch(share=False, debug=True, mcp_server=True)
    logging.info("Gradio interface and MCP server launched. Access at the URL above.")
