# Hugging Face PR Tagging Agent

This project is an automated agent for managing tags on Hugging Face Hub repositories. It listens for discussion comment webhooks, extracts relevant tags from the comments or discussion titles, and uses the Hugging Face Hub API to add missing tags to the repository by creating pull requests. The agent leverages the MCP (Multi-Command Protocol) and integrates with FastAPI for webhook handling and Gradio for manual testing.

## Features

- **Automatic Tag Extraction:** Detects and extracts ML/AI-related tags from discussion comments and titles using natural language processing and pattern matching.
- **Webhook Listener:** Receives and validates Hugging Face Hub webhook events securely.
- **Automated PR Creation:** Adds new tags to repository metadata by creating pull requests to update the `README.md` (Model Card).
- **MCP Server Integration:** Uses a subprocess MCP server to expose tools for tag management.
- **Gradio UI:** Provides a simple web interface for manually testing tag retrieval and addition.
- **Environment-based Configuration:** Supports `.env` files for local development and secure token management.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/etheal9/hf-pr-agent.git
cd hf-pr-agent
pip install -r requirements.txt
```

