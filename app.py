
import os
import json
import re
import logging
from typing import Optional, Literal, Dict, Any, List

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from dotenv import load_dotenv # Used for local development to load .env file

# Import the Agent class for MCP client capabilities
from huggingface_hub.inference._mcp.agent import Agent

# Configure basic logging for the FastAPI app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file for local development.
# This ensures that HF_TOKEN and WEBHOOK_SECRET are available.
load_dotenv()

# --- Global Configuration ---
HF_TOKEN = os.getenv("HF_TOKEN")
# Default model for the agent. Can be overridden by HF_MODEL env var.
HF_MODEL = os.getenv("HF_MODEL", "microsoft/DialoGPT-medium")
# Provider for the agent's language model. "hf-inference" uses Hugging Face Inference API.
DEFAULT_PROVIDER: Literal["hf-inference"] = "hf-inference"
# Webhook secret for authenticating incoming Hugging Face webhooks
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

# Global agent instance to ensure it's created only once (singleton pattern)
agent_instance: Optional[Agent] = None

# --- FastAPI App Initialization ---
app = FastAPI(title="Hugging Face PR Tagging Agent", version="0.1.0")

# --- Agent Management Function ---
async def get_agent() -> Optional[Agent]:
    """
    Get or create a singleton Agent instance.
    The agent connects to the mcp_server.py via stdio (subprocess).
    """
    logging.info("🤖 get_agent() called...")
    global agent_instance
    
    if agent_instance is None:
        if not HF_TOKEN:
            logging.error("❌ HF_TOKEN environment variable not set. Cannot create Agent instance.")
            return None # Agent cannot be created without a token

        logging.info("🔧 Creating new Agent instance...")
        logging.info(f"🔑 HF_TOKEN present: {bool(HF_TOKEN)}")
        logging.info(f"🤖 Model: {HF_MODEL}")
        logging.info(f"🔗 Provider: {DEFAULT_PROVIDER}")
        
        try:
            agent_instance = Agent(
                model=HF_MODEL,
                provider=DEFAULT_PROVIDER,
                api_key=HF_TOKEN,
                servers=[
                    {
                        "type": "stdio", # Connect via standard input/output
                        "config": {
                            "command": "python", # Command to run the MCP server
                            "args": ["mcp_server.py"], # The MCP server script
                            "cwd": ".", # Current working directory for the subprocess
                            "env": {"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {}, # Pass HF_TOKEN to server subprocess
                        },
                    }
                ],
            )
            logging.info("✅ Agent instance created successfully")
            logging.info("🔧 Loading tools...")
            await agent_instance.load_tools() # Discover and load tools from the MCP server
            logging.info("✅ Tools loaded successfully")
        except Exception as e:
            logging.error(f"❌ Error creating/loading agent: {str(e)}")
            logging.exception("Detailed traceback for agent creation error:") # Log full traceback
            agent_instance = None # Reset agent_instance on failure

    return agent_instance

# --- Tag Extraction Logic ---
# Recognized ML/AI tags for validation and natural language extraction
RECOGNIZED_TAGS = {
    "pytorch", "tensorflow", "jax", "transformers", "diffusers",
    "text-generation", "text-classification", "question-answering",
    "text-to-image", "image-classification", "object-detection",
    "fill-mask", "token-classification", "translation", "summarization",
    "feature-extraction", "sentence-similarity", "zero-shot-classification",
    "image-to-text", "automatic-speech-recognition", "audio-classification",
    "voice-activity-detection", "depth-estimation", "image-segmentation",
    "video-classification", "reinforcement-learning", "tabular-classification",
    "tabular-regression", "time-series-forecasting", "graph-ml", "robotics",
    "computer-vision", "nlp", "cv", "multimodal",
    # Add more as needed based on common HF Hub tags
}

def extract_tags_from_text(text: str) -> List[str]:
    """
    Extract potential tags from discussion text using multiple patterns.
    Filters tags against a list of recognized ML/AI tags.
    """
    text_lower = text.lower()
    explicit_tags = []
    
    logging.debug(f"Extracting tags from: {text_lower}")

    # Pattern 1: "tag: something" or "tags: something, another"
    tag_pattern = r"tags?:\s*([a-zA-Z0-9-_\s,]+)"
    matches = re.findall(tag_pattern, text_lower)
    for match in matches:
        tags = [tag.strip() for tag in match.split(",") if tag.strip()]
        explicit_tags.extend(tags)
    logging.debug(f"Explicit tags (Pattern 1): {explicit_tags}")

    # Pattern 2: "#hashtag" style
    hashtag_pattern = r"#([a-zA-Z0-9-_]+)"
    hashtag_matches = re.findall(hashtag_pattern, text_lower)
    explicit_tags.extend(hashtag_matches)
    logging.debug(f"Explicit tags (Pattern 2 - hashtags): {explicit_tags}")


    # Pattern 3: Look for recognized tags mentioned in natural text
    mentioned_tags = []
    for tag in RECOGNIZED_TAGS:
        # Use word boundaries to avoid partial matches (e.g., "nlp" matching "nlptask")
        if re.search(r'\b' + re.escape(tag) + r'\b', text_lower):
            mentioned_tags.append(tag)
    logging.debug(f"Mentioned tags (Pattern 3): {mentioned_tags}")

    # Combine and deduplicate all found tags
    all_tags = list(set(explicit_tags + mentioned_tags))
    logging.debug(f"All unique tags before final filtering: {all_tags}")

    # Final filtering: Only include tags from RECOGNIZED_TAGS or those explicitly mentioned.
    # This prevents arbitrary strings from being added as tags.
    valid_tags = []
    for tag in all_tags:
        if tag in RECOGNIZED_TAGS or tag in explicit_tags: # explicit_tags covers pattern 1 & 2
            valid_tags.append(tag)
    
    logging.info(f"🔍 Extracted and validated tags: {valid_tags}")
    return valid_tags

# --- Webhook Processing Logic ---
async def process_webhook_comment(webhook_data: Dict[str, Any]):
    """
    Processes a Hugging Face Hub discussion comment webhook event.
    Extracts tags and uses the MCP agent to add them to the repository.
    This function is designed to be run as a background task.
    """
    logging.info("🏷️ Starting process_webhook_comment (background task)...")

    try:
        # Extract relevant information from the webhook payload
        comment_content = webhook_data.get("comment", {}).get("content", "")
        discussion_title = webhook_data.get("discussion", {}).get("title", "")
        repo_name = webhook_data.get("repo", {}).get("name", "")
        
        if not repo_name:
            logging.error("❌ Webhook data missing 'repo.name'. Cannot process.")
            return

        logging.info(f"Processing comment for repo: '{repo_name}' - Discussion: '{discussion_title}'")
        logging.debug(f"Comment content: {comment_content}")

        # Extract potential tags from the comment and discussion title
        comment_tags = extract_tags_from_text(comment_content)
        title_tags = extract_tags_from_text(discussion_title)
        
        # Combine and deduplicate all extracted tags
        # Using a set directly here would be fine too, then convert to list.
        all_potential_tags = list(set(comment_tags + title_tags))

        logging.info(f"🔍 All unique potential tags extracted: {all_potential_tags}")

        if not all_potential_tags:
            logging.info(f"No recognizable tags found in the discussion for {repo_name}. Skipping.")
            return ["No recognizable tags found in the discussion."]

        # Get agent instance (will create if not exists)
        agent = await get_agent()
        if not agent:
            logging.error(f"Agent not configured (missing HF_TOKEN or initialization failed). Cannot process tags for {repo_name}.")
            return ["Error: Agent not configured (missing HF_TOKEN or initialization failed)"]

        # Process each tag using the agent
        result_messages = []
        for tag in all_potential_tags:
            try:
                # The prompt guides the agent on what to do with the tag
                prompt = f"""
                For the Hugging Face repository '{repo_name}', perform the following actions:
                1. Check the current tags of the repository.
                2. If the tag '{tag}' is not already present, add it to the repository's metadata (README.md)
                   by creating a pull request.
                3. Provide a summary of the action taken (e.g., tag added, tag already exists, error).

                Repository: {repo_name}
                Tag to check/add: {tag}
                """
                
                logging.info(f"🤖 Agent processing tag '{tag}' for repo '{repo_name}' with prompt...")
                response = await agent.run(prompt) # Agent intelligently calls get_current_tags then add_new_tag

                # The agent's response is a natural language summary. Parse it for status.
                response_lower = response.lower()
                if "success" in response_lower or "created pr" in response_lower or "already exists" in response_lower:
                    # Look for positive indicators, or if tag already exists (which is also a "success" for the goal)
                    result_messages.append(f"✅ Tag '{tag}' processed for '{repo_name}': {response.strip()}")
                    logging.info(f"Agent response for '{tag}': {response.strip()}")
                else:
                    # Assume failure if no clear success indicator
                    result_messages.append(f"⚠️ Issue with tag '{tag}' for '{repo_name}': {response.strip()}")
                    logging.warning(f"Agent response (potential issue) for '{tag}': {response.strip()}")
                    
            except Exception as e:
                error_msg = f"❌ Error during agent processing for tag '{tag}' in '{repo_name}': {str(e)}"
                logging.error(error_msg)
                logging.exception("Detailed traceback for agent run error:")
                result_messages.append(error_msg)

        logging.info(f"Finished processing tags for {repo_name}. Results: {result_messages}")
        return result_messages

    except Exception as e:
        logging.error(f"❌ General error in process_webhook_comment: {str(e)}")
        logging.exception("Detailed traceback for webhook comment processing error:")
        return [f"General processing error: {str(e)}"]

# --- FastAPI Endpoints ---

@app.get("/")
async def root():
    """Simple root endpoint for health checks."""
    logging.info("Received request on root endpoint.")
    return {"message": "Hugging Face PR Tagging Agent is running!"}

@app.post("/webhook")
async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
    """
    Handles incoming Hugging Face Hub webhook events.
    Validates the secret and processes the event in a background task.
    """
    logging.info("🔔 Received webhook request.")

    # 1. Validate Webhook Secret for security
    if not WEBHOOK_SECRET:
        logging.error("❌ WEBHOOK_SECRET environment variable not set. Cannot validate webhooks.")
        raise HTTPException(status_code=500, detail="Server configuration error: Webhook secret not set.")

    # Hugging Face webhooks often send the secret in the 'X-Webhook-Secret' header
    x_webhook_secret = request.headers.get("X-Webhook-Secret")

    if x_webhook_secret != WEBHOOK_SECRET:
        logging.warning("🚨 Invalid webhook secret received. Request denied.")
        raise HTTPException(status_code=401, detail="Invalid webhook secret")
    
    try:
        webhook_data = await request.json()
        logging.info(f"Webhook data received (parsed as JSON). Event type: {webhook_data.get('event', 'N/A')}")
        logging.debug(f"Full webhook payload: {json.dumps(webhook_data, indent=2)}")

        # Check if it's a discussion comment event
        if webhook_data.get("event") == "discussion" and webhook_data.get("action") == "comment":
            # Process the webhook in a background task to return a quick response
            background_tasks.add_task(process_webhook_comment, webhook_data)
            logging.info("✅ Webhook accepted and processing initiated in background.")
            return {"status": "accepted", "message": "Webhook received and processing in background."}
        else:
            logging.info(f"Webhook event type '{webhook_data.get('event', 'N/A')}' or action '{webhook_data.get('action', 'N/A')}' not handled. Skipping.")
            return {"status": "ignored", "message": "Webhook event not relevant for tagging agent."}

    except json.JSONDecodeError:
        logging.error("❌ Received webhook with invalid JSON payload.")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        logging.error(f"❌ Error processing webhook: {str(e)}")
        logging.exception("Detailed traceback for webhook handler error:")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- Example Tool Usage (for testing the agent directly if needed) ---
# You can uncomment and run this function directly in a script or a test environment
# to see the agent in action without a full webhook flow.
# async def example_tool_usage():
#     """
#     Example of how the agent would use tools based on a natural language prompt.
#     """
#     logging.info("\n--- Running example_tool_usage ---")
#     agent = await get_agent()
    
#     if agent:
#         try:
#             # The agent can reason about which tools to use
#             prompt = "Check the current tags for 'HuggingFaceH4/zephyr-7b-alpha' and add the tag 'large-language-model' if it's not already present."
#             logging.info(f"Sending prompt to agent: '{prompt}'")
#             response = await agent.run(prompt)
#             logging.info(f"Agent response: {response}")
#         except Exception as e:
#             logging.error(f"Error during example tool usage: {str(e)}")
#             logging.exception("Detailed traceback for example tool usage error:")
#     else:
#         logging.error("Agent not available for example tool usage.")

# To run the example (for local testing, can be placed at the end of the file or in a separate script):
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(example_tool_usage())