
# 11KU7 AI Coder (version 1.0)
# Intelligent Assistance, Right in Your Terminal

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)


![11ku7-logo-110325_imresizer](https://github.com/user-attachments/assets/6b23d280-274f-44df-b939-df06455614af)


**11KU7 AI Coder** is a robust yet remarkably user-friendly AI coding assistant, meticulously crafted in Python to revolutionize your software development process.  This command-line powerhouse seamlessly integrates a diverse array of cutting-edge Large Language Models (LLMs) and specialized AI tools, putting a wealth of intelligent capabilities directly at your fingertips.  Imagine effortlessly generating clean, efficient code across multiple programming languages, intelligently managing complex development projects, quickly deciphering and extracting insights from intricate documents, and performing lightning-fast web research – all without ever leaving the comfort and efficiency of your terminal.  

**11KU7 AI Coder** is designed to be your indispensable development companion, streamlining every stage of your workflow from initial concept to final execution, boosting your productivity and empowering you to code smarter, not harder.

## ✨ Key Features

*   **Multi-Model Support:**  Choose from a wide range of AI model providers including Groq, OpenRouter, Ollama, OpenAI, Gemini, Mistral, Anthropic, and more!
*   **Versatile Modes:**  Explore 11 distinct modes for different development tasks:
    *   **Code Generation Mode (`code`):**  Generate code snippets in various programming languages.
    *   **Web App Generation Mode (`web app`):**  Create basic web applications with HTML, JavaScript, and Tailwind CSS.
    *   **Ask Code Mode (`ask code`):** Analyze and answer questions about your code files.
    *   **Ask Image Mode (`ask image`):**  Understand and query image content (local files or URLs).
    *   **Ask URL Mode (`ask url`):**  Summarize and get insights from webpage content.
    *   **Ask Document Mode (`ask docu`):**  Extract information and answer questions from PDF, Markdown, and text documents.
    *   **Image Generation Mode (`image`):**  Generate images from text prompts using Hugging Face models.
    *   **Shell Agent Mode (`shell agent`):**  Get intelligent assistance with shell commands for your OS.
    *   **Vitex App Mode (`vitex app`):**  Interactive mode to create and modify React apps with Vite or Expo, including running and previewing.
    *   **Git Repository Mode (`gitrepo`):**  Manage and modify general Git repositories, with code understanding and command suggestions.
    *   **Web Search Mode (`web search`):**  Perform web searches and get summarized answers based on search results using Brave Search API.
*   **Interactive Menu:**  Easy-to-navigate command-line menu to access all features and modes.
*   **Conversation History:** Save your AI interactions to markdown files for future reference.
*   **File Management:** Basic file and folder creation, directory navigation, and file execution directly from the tool.
*   **Cross-Platform Compatibility:**  Designed to work on Windows, Linux, macOS, and Termux (Android).
*   **Optional Dependencies & Installation Help:**  Provides guidance for installing optional tools like Poppler for advanced PDF features.

## 🚀 Getting Started

### Prerequisites

*   **Python 3.7+:**  Make sure you have Python 3.7 or higher installed on your system.
*   **API Keys:** You'll need API keys for the AI model providers you intend to use (e.g., OpenAI, Groq, OpenRouter, Mistral, etc.).  Brave Search API key is optional but recommended for Web Search Mode. Hugging Face API Key is needed for Image Generation.
*   **Optional Dependencies:**
    *   **Poppler (Windows):** For advanced image querying in PDF documents in "ask docu" mode (installation instructions provided in the tool via `install poppler` or `install help`).
    *   **Node.js and npm (for Vitex App Mode):** Required for creating and running React/Expo applications.
    *   **Git (for Git Repository Mode):** Required for working with Git repositories.


## 📦 Installation


### Clone the repository
```bash
git clone https://github.com/dheeraj21/11ku7-ai-coder.git
```

### Change the directory
```bash
cd 11ku7-ai-coder
```

### Install required libraries
```bash
pip install -r requirements.txt
```

### Running the Script
```bash
python 11ku7-ai-coder-v-1-0.py
```

Follow the on-screen prompts to select your model provider, configure API keys, and start using the AI Coder!

### Running in windows with executable

You can directly run 11ku7-ai-coder by running executable in windows.

[11ku7-ai-coder-v-1-0-windows](https://github.com/dheeraj21/11ku7-ai-coder/releases/download/11ku7-ai-coder-v-1-0-windows/11ku7-ai-coder-v-1-0-windows.exe)


## 📝 Usage

Once you run the script, you'll be presented with a menu outlining the available commands and modes.

*   **Type commands** directly at the prompt to activate modes or perform actions (e.g., `code`, `web app`, `ask code`, `shell`, `help`, `quit`).
*   **Use `help` command** to display the full list of available commands and modes within the tool.
*   **Follow prompts** within each mode to provide queries, file paths, and other necessary information.
*   **Use `op` command** to change the model provider and model in the current session.
*   **Use `save` command** within most modes to save your conversation history to a markdown file.


## 🧰 Modes in Detail

Break down of each of the 11 modes in detail:

**1. code (Code Generation Mode):**

*   **Purpose:** To generate code snippets in various programming languages based on user prompts.
*   **Activation:** Type `code` at the main prompt.
*   **Functionality:**
    *   Takes natural language queries describing code requirements (e.g., "generate a python function to calculate factorial").
    *   Utilizes the selected AI model (via `code_generator` agent) to generate code.
    *   Displays the generated code in a markdown code block with syntax highlighting.
    *   Offers to save the generated code to a file.
    *   Allows saving the entire conversation history.
*   **Input:** Natural language code requests.
*   **Output:** Code snippets, typically in markdown format.
*   **Underlying Function:** `code()`

**2. web app (Web App Generation Mode):**

*   **Purpose:** To generate basic web applications (HTML and JavaScript) based on user descriptions.
*   **Activation:** Type `web app` at the main prompt.
*   **Functionality:**
    *   Takes natural language descriptions of web apps (e.g., "create a simple to-do list web app").
    *   Uses the selected AI model (via `web_app_generator` agent) to generate HTML and JavaScript code.
    *   Employs Tailwind CSS for styling (inline in HTML).
    *   Displays HTML and JavaScript code in separate markdown code blocks.
    *   Offers to save the code to an HTML file.
    *   Optionally runs a local Python HTTP server to preview the web app.
    *   Allows saving the conversation history.
*   **Input:** Web app descriptions in natural language.
*   **Output:** HTML and JavaScript code snippets in markdown format.
*   **Underlying Function:** `web_app()`

**3. ask code (Ask Code Mode):**

*   **Purpose:** To analyze and answer questions about existing code files.
*   **Activation:** Type `ask code` at the main prompt.
*   **Functionality:**
    *   Prompts for the path to a code file.
    *   Reads the content of the specified code file.
    *   Takes natural language questions about the code (e.g., "explain this function", "what does this code do?").
    *   Uses the selected AI model to answer questions based on the code content.
    *   Displays the AI's response in markdown.
    *   Offers to save the conversation history.
    *   Includes a shell sub-mode for command based operations.
*   **Input:** File path to a code file, questions about the code in natural language.
*   **Output:** AI's answers and explanations about the code, in markdown format.
*   **Underlying Function:** `ask_code_mode()`

**4. ask image (Ask Image Mode):**

*   **Purpose:** To analyze and answer questions about images, either from local files or URLs.
*   **Activation:** Type `ask image` at the main prompt.
*   **Functionality:**
    *   Prompts for the path to an image file (local or URL).
    *   Encodes the image into base64 or uses the URL.
    *   Takes natural language questions about the image (e.g., "what objects are in this image?", "describe the scene").
    *   Uses a vision-capable AI model to answer questions based on the image content.
    *   Displays the AI's response in markdown.
    *   Offers to save the conversation history.
    *   Includes a shell sub-mode for command based operations.
*   **Input:** File path or URL of an image, questions about the image in natural language.
*   **Output:** AI's descriptions and answers about the image, in markdown format.
*   **Underlying Function:** `ask_the_image_mode()`

**5. ask url (Ask URL Mode):**

*   **Purpose:** To analyze and answer questions about the content of a webpage specified by a URL.
*   **Activation:** Type `ask url` at the main prompt.
*   **Functionality:**
    *   Prompts for a URL.
    *   Fetches the content of the webpage at the given URL using web scraping.
    *   Takes natural language questions about the webpage content (e.g., "summarize this page", "what is the main topic?").
    *   Uses the selected AI model to answer questions based on the scraped webpage content.
    *   Displays the AI's response in markdown.
    *   Offers to save the conversation history.
*   **Input:** URL of a webpage, questions about the page content in natural language.
*   **Output:** AI's summaries and answers about the webpage, in markdown format.
*   **Underlying Function:** `ask_url_mode()`

**6. ask docu (Ask Document Mode):**

*   **Purpose:** To analyze and answer questions about the content of document files (PDF, Markdown, or text files).
*   **Activation:** Type `ask docu` at the main prompt.
*   **Functionality:**
    *   Prompts for the path to a document file.
    *   Reads the text content from the document (and optionally extracts images from PDFs using Poppler if installed).
    *   Takes natural language questions about the document content (e.g., "what is the document about?", "summarize page 3").
    *   Uses the selected AI model to answer questions based on the document content (and optionally images).
    *   Displays the AI's response in markdown.
    *   Offers to save the conversation history.
    *   Includes a shell sub-mode for command based operations.
*   **Input:** File path to a document file (PDF, MD, TXT), questions about the document in natural language.
*   **Output:** AI's summaries and answers about the document, in markdown format.
*   **Underlying Function:** `ask_document_mode()`

**7. image (Image Generation Mode):**

*   **Purpose:** To generate images based on text prompts using a selected image generation model (Hugging Face).
*   **Activation:** Type `image` at the main prompt.
*   **Functionality:**
    *   Prompts for a text description of the desired image.
    *   Uses the selected image generation model (via Hugging Face Inference Client and `IMAGE_MODEL`) to generate an image.
    *   Saves the generated image to a file (PNG, JPG, or JPEG).
    *   Displays the save path and generation time.
    *   Allows generating multiple images in a session.
*   **Input:** Text prompts describing the desired image.
*   **Output:** Saved image files (PNG, JPG, or JPEG).
*   **Underlying Function:** `image()`

**8. shell agent (Shell Agent Mode):**

*   **Purpose:** To assist in generating and executing shell commands for the operating system.
*   **Activation:** Type `shell agent` at the main prompt.
*   **Functionality:**
    *   Takes natural language descriptions of shell tasks (e.g., "list all files in the current directory", "create a backup of my documents").
    *   Uses the `shell_agent` to generate shell commands appropriate for the detected operating system (Windows, Linux, macOS, Termux).
    *   Displays the generated commands.
    *   Allows executing the commands step-by-step or all at once.
    *   Shows the output of executed commands.
    *   Maintains a history of executed commands.
    *   Offers to save the conversation history (including commands and outputs).
*   **Input:** Shell tasks described in natural language.
*   **Output:** Shell commands, output of executed commands.
*   **Underlying Function:** `shell_agent_mode()`

**9. vitex app (Vitex App Mode):**

*   **Purpose:** To assist in creating and modifying React applications using Vite or Expo frameworks. This mode is interactive and helps in project setup, code modification, running, and previewing web/mobile applications.
*   **Activation:** Type `vitex app` at the main prompt.
*   **Functionality:**
    *   Allows creating new Vite/React or Expo projects from scratch.
    *   Allows working with existing projects (Vite/React or Expo).
    *   Ingests project file structure and code.
    *   Takes natural language requests to modify the app (e.g., "add a button to change background color", "create a new component").
    *   Uses an AI agent to generate modification plans and code changes.
    *   Offers to preview code changes and approve before applying.
    *   Applies code changes to files, backs up files before modification, and can revert changes.
    *   Suggests and executes necessary commands (like `npm install`).
    *   Allows running and previewing the app (web or mobile).
    *   Includes a chat sub-mode for asking questions about the project.
    *   Includes a shell sub-mode for command based operations.
    *   Offers to save conversation history.
*   **Input:** Natural language requests to create or modify React/Expo applications.
*   **Output:** Modified code files, executed commands, previews of running applications.
*   **Underlying Function:** `vitex_app_mode()`

**10. gitrepo (Git Repository Mode):**

*   **Purpose:** To assist in managing and modifying general Git repositories (not limited to web apps, can be any code repository). This mode focuses on repository-level tasks and code modifications within a Git repository context.
*   **Activation:** Type `gitrepo` at the main prompt.
*   **Functionality:**
    *   Allows creating new Git repositories.
    *   Allows cloning existing repositories from URLs.
    *   Allows working with local existing repositories.
    *   Ingests repository file structure and code.
    *   Detects the main/entry file of the repository.
    *   Takes natural language requests to modify the repository or code (e.g., "add a new feature", "fix a bug", "update documentation").
    *   Uses an AI agent to generate modification plans and code changes.
    *   Offers to preview code changes and approve before applying.
    *   Applies code changes to files, backs up files before modification, and can revert changes.
    *   Suggests and executes necessary commands (like dependency installations).
    *   Includes a chat sub-mode for asking questions about the repository.
    *   Includes a shell sub-mode for command based operations.
    *   Offers to save conversation history.
*   **Input:** Natural language requests to manage or modify Git repositories.
*   **Output:** Modified code files, executed commands, repository structure previews.
*   **Underlying Function:** `gitrepo_mode()`

**11. web search (Web Search Mode):**

*   **Purpose:** To perform web searches using the Brave Search API and provide answers based on the search results.
*   **Activation:** Type `web search` at the main prompt.
*   **Functionality:**
    *   Takes natural language search queries.
    *   Uses the Brave Search API to perform web searches.
    *   Scrapes content from the top search result webpages (limited to the first few results).
    *   Uses an AI agent to summarize the search results and answer the user's query based on the web content.
    *   Cites sources (URLs) inline in the response.
    *   Displays the AI's response in markdown.
    *   Requires a Brave Search API key to function properly.
    *   Offers to save the conversation history.
*   **Input:** Search queries in natural language.
*   **Output:** Summarized answers based on web search results, with source citations, in markdown format.
*   **Underlying Function:** `web_search_mode()`

These are the 11 modes available in the script, each designed for a specific type of task related to coding, development, and information retrieval using AI. They collectively make the `11KU7 AI CODER` a versatile tool for AI-assisted programming and related activities.

## 🤝 Contributing

Whole purpose of opensourcing this code are contributions. If you have ideas for new features, improvements, or bug fixes, please feel free to:

1.  **Fork** the repository.
2.  **Create a branch** for your feature or fix.
3.  **Commit** your changes.
4.  **Push** to your branch.
5.  **Open a Pull Request.**

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## 📧 Contact & Support

For questions, issues, or feedback, please feel free to connect


I'm always open to collaborations! Feel free to reach out:

[![X](https://img.shields.io/badge/X-%40er_dheeraj15-blue?style=flat-square&logo=X&logoColor=white)](https://x.com/er_dheeraj15)


## 🙏 Acknowledgements

* The concept of Agents, as implemented in this AI coder, is inspired by and builds upon the principles of modular AI design and agent-based systems. Specifically, the approach to creating agents from scratch, 
  without relying on LLM frameworks, is acknowledged to be influenced by the ideas presented in the following Research paper:

  **REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS** [here](https://arxiv.org/pdf/2210.03629) 

  This research paper provided valuable insights into building agents, which has informed the design and implementation of the `Agent` class and specialized agents within this project. I appreciate all authors 
  for sharing their knowledge and contributing to the understanding of agent-based AI systems.


*  The concept of use of ingestion of file structure of whole repository was taken from the source code of this project 

   **cyclotruc/gitingest** [here](https://github.com/cyclotruc/gitingest)

   This repository's source code provided an important base for file ingestion to create project states in vitex and gitrepo mode of 11ku7-ai-coder. 



