import json
import subprocess
import os
import re
from rich import print
from rich.panel import Panel
from rich.console import Console
from rich.columns import Columns
from rich.text import Text
from rich.theme import Theme
from rich.prompt import Prompt
import openai
import threading
import time
import signal
from rich.syntax import Syntax
import requests

from module import generate_11ku7_cover_page, display_11ku7_logo_green

from datetime import datetime, timezone
import base64
from fnmatch import fnmatch
from bs4 import BeautifulSoup

if os.name == 'nt':
    import msvcrt
else:
    import sys, tty, termios

# Custom theme
custom_theme = Theme({
    "info": "white",
    "success": "green",
    "warning": "red",
})
console = Console(theme=custom_theme)

def get_operating_system():
    if os.name == 'nt':
        return "Windows"
    else:
        if os.path.isfile("/data/data/com.termux/files/usr/bin/login"):
            return "Termux"
        try:
            with open('/proc/version', 'r') as version_file:
                if "Android" in version_file.read():
                    return "Termux"
        except IOError:
            pass
        if sys.platform.startswith('linux'):
            return "Linux"
        elif sys.platform == 'darwin':
            return "MacOS"
        else:
            return "Unix"

operating_system = get_operating_system()

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_masked_input(prompt):
    if os.name == 'nt':
        console.print(prompt, end='')
        password = ''
        while True:
            char = msvcrt.getwch()
            if char in ('\r', '\n'):
                print()
                return password
            elif char == '\b':
                if password:
                    password = password[:-1]
                    print('\b \b', end='', flush=True)
            else:
                password += char
                print('*', end='', flush=True)
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            console.print(prompt, end='')
            password = ''
            while True:
                char = sys.stdin.read(1)
                if char in ('\r', '\n'):
                    print()
                    return password
                elif char == '\x7f':
                    if password:
                        password = password[:-1]
                        print('\b \b', end='', flush=True)
                else:
                    password += char
                    print('*', end='', flush=True)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

start_time = datetime.now(timezone.utc)

def select_model():
    provider_menu = """
    1. Groq
    2. OpenRouter
    3. Ollama
    4. llama.cpp
    5. Github
    6. Grok
    7. Glhf
    8. Anthropic
    9. Openai
    10. Gemini
    11. Mistral
    12. Huggingface
    13. DeepSeek
    14. Hyperbolic
    15. Sambanova
    16. Together.ai
    """
    console.print(Panel(provider_menu, title="[bold]Select a model provider[/bold]"))
    while True:
        choice = Prompt.ask("Enter the number of your chosen model provider")
        if choice in [str(i) for i in range(1, 17)]:
            provider_options = {
                "1": ("Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
                "2": ("OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
                "3": ("Ollama", "http://localhost:11434/v1", "OLLAMA_API_KEY"),
                "4": ("llama.cpp", "http://localhost:8080/v1", "LLAMA_CPP_API_KEY"),
                "5": ("Github", "https://models.inference.ai.azure.com", "GITHUB_TOKEN"),
                "6": ("Grok", "https://api.x.ai/v1", "XAI_GROK_API_KEY"),
                "7": ("Glhf", "https://glhf.chat/api/openai/v1", "GLHF_API_KEY"),
                "8": ("Anthropic", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
                "9": ("Openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
                "10": ("Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
                "11": ("Mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
                "12": ("Huggingface", "", "HUGGINGFACE_API_KEY"),
                "13": ("DeepSeek", "https://api.deepseek.com/v1", "DeepSeek API Key"),
                "14": ("Hyperbolic", "https://api.hyperbolic.xyz/v1", "HYPERBOLIC_API_KEY"),
                "15": ("Sambanova", "https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY"),
                "16": ("Together.ai", "https://api.together.xyz/v1", "TOGETHER_API_KEY")
            }
            return provider_options[choice]
        console.print("[warning]Invalid choice. Please try again.[/warning]")

def get_api_key(provider, env_var_name):
    if provider in ["Ollama", "llama.cpp"]:
        return "ollama" if provider == "Ollama" else "llama.cpp"
    if provider == "Huggingface":
        while True:
            console.print(Panel(f"[bold]Enter {provider} API Key[/bold]", title="API Key Input"))
            api_key = get_masked_input(f"Enter {provider} API Key (input will be masked): ")
            if not api_key.strip():
                console.print("[warning]API key cannot be empty. Please enter a valid API key.[/warning]")
                continue
            os.environ[env_var_name] = api_key
            console.print(Panel(f"[bold]Enter {provider} endpoint (Optional)[/bold]", title="Endpoint Input"))
            base_url = input(f"Enter {provider} endpoint (press enter to skip): ")
            return api_key, base_url
    while True:
        console.print(Panel(f"[bold]Enter {provider} API Key[/bold]", title="API Key Input"))
        api_key = get_masked_input(f"Enter {provider} API Key (input will be masked): ")
        if not api_key.strip():
            console.print("[warning]API key cannot be empty. Please enter a valid API key.[/warning]")
            continue
        os.environ[env_var_name] = api_key
        return api_key


generate_11ku7_cover_page()

clear_console()

display_11ku7_logo_green()


provider, base_url, env_var_name = select_model()
console.print(Panel(f"[bold]Provider: {provider}[/bold]", title="Selected Provider"))
os_panel = Panel(f"[bold]Operating System: {operating_system}[/bold]", title="System Information")

if provider == "llama.cpp":
    try:
        response = requests.get("http://localhost:8080/v1/models", timeout=5)
        response.raise_for_status()
        model_name = response.json()["data"][0]["id"]
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error: Could not connect to llama.cpp server. Error: {e}[/red]")
        model_name = "llama.cpp server not running"
elif provider == "Ollama":
    try:
        response = requests.get("http://localhost:11434/v1/models", timeout=5)
        response.raise_for_status()
        data = response.json()
        if "models" in data and data["models"]:
            console.print(Panel("[bold]Available Ollama Models[/bold]", title="Model Selection"))
            for idx, model in enumerate(data["models"], 1):
                console.print(f"[bold white]{idx}.[/bold white] {model['name']}")
            while True:
                model_choice = Prompt.ask("Select the number of the Ollama model")
                try:
                    model_index = int(model_choice) - 1
                    if 0 <= model_index < len(data["models"]):
                        model_name = data["models"][model_index]["name"]
                        break
                    console.print("[warning]Invalid choice.[/warning]")
                except ValueError:
                    console.print("[warning]Invalid choice. Enter a number.[/warning]")
        elif "data" in data and data["data"]:
            console.print(Panel("[bold]Available Ollama Models[/bold]", title="Model Selection"))
            for idx, model in enumerate(data["data"], 1):
                console.print(f"[bold white]{idx}.[/bold white] {model['id']}")
            while True:
                model_choice = Prompt.ask("Select the number of the Ollama model")
                try:
                    model_index = int(model_choice) - 1
                    if 0 <= model_index < len(data["data"]):
                        model_name = data["data"][model_index]["id"]
                        break
                    console.print("[warning]Invalid choice.[/warning]")
                except ValueError:
                    console.print("[warning]Invalid choice. Enter a number.[/warning]")
        else:
            console.print("[red]Error: No models found from Ollama server.[/red]")
            model_name = "ollama server no models found"
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error: Could not connect to Ollama server. Error: {e}[/red]")
        model_name = "ollama server not running"
else:
    console.print(Panel("[bold]Enter model name[/bold]", title="Model Configuration"))
    model_name = Prompt.ask("Model name", default={
        "Groq": "llama-3.3-70b-versatile",
        "OpenRouter": "deepseek/deepseek-r1-zero:free",
        "Ollama": "local model",
        "llama.cpp": "local model",
        "Github": "gpt-4o",
        "Grok": "grok-2-latest",
        "Glhf": "hf:Qwen/Qwen2.5-Coder-32B-Instruct",
        "Anthropic": "claude-3-5-sonnet-20241022",
        "Openai": "gpt-4o",
        "Gemini": "gemini-2.0-flash-thinking-exp-01-21",
        "Mistral": "mistral-large-latest",
        "Huggingface": "meta-llama/Meta-Llama-3-8B-Instruct",
        "DeepSeek": "deepseek-chat",
        "Hyperbolic": "meta-llama/Llama-3.3-70B-Instruct",
        "Sambanova": "Meta-Llama-3.3-70B-Instruct",
        "Together.ai": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    }.get(provider, "default-model"))

if provider == "Glhf" and model_name != "hf:Qwen/Qwen2.5-Coder-32B-Instruct":
    model_name = "hf:" + model_name

if provider == "Huggingface":
    api_key, base_url = get_api_key(provider, env_var_name)
else:
    api_key = get_api_key(provider, env_var_name)

ROUTING_MODEL = TOOL_USE_MODEL = GENERAL_MODEL = model_name

console.print(Panel("[bold]Select Image Generation Model Provider[/bold]", title="Image Model Provider"))
image_provider_menu = """1. Huggingface"""
console.print(Panel(image_provider_menu, title="[bold]Image Model Provider Menu[/bold]"))
while True:
    image_provider_choice = Prompt.ask("Enter the number of your chosen image model provider", default="1")
    if image_provider_choice == "1":
        IMAGE_PROVIDER = "Huggingface"
        break
    console.print("[warning]Invalid choice. Please try again.[/warning]")

console.print(Panel("[bold]Enter Image Generation model name[/bold]", title="Image Model Configuration"))
image_model_name = Prompt.ask("Image Generation model name", default="black-forest-labs/FLUX.1-dev")

if IMAGE_PROVIDER == "Huggingface":
    console.print(Panel(f"[bold]Enter Huggingface API Key (Leaving it blank sets Image Generation: off)[/bold]", title="Image Generation Model API Key Input"))
    image_api_key = get_masked_input(f"Enter Huggingface API Key (input will be masked): ")
    if not image_api_key.strip():
        console.print("[warning]API key not entered. Image Generation sets to off.[/warning]")
else:
    image_api_key = None

IMAGE_MODEL = image_model_name

if provider == "Anthropic":
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
elif provider == "Mistral":
    from mistralai import Mistral
    client = Mistral(api_key=api_key)
elif provider == "Huggingface":
    from huggingface_hub import InferenceClient
    client = InferenceClient(base_url=base_url, api_key=api_key)
else:
    client = openai.OpenAI(base_url=base_url, api_key=api_key)

# Brave Search Function (new addition)
def brave_search(query: str, brave_api_key: str, num_results: int = 5) -> list:
    headers = {
        "X-Subscription-Token": brave_api_key,
        "Accept": "application/json",
    }
    url = "https://api.search.brave.com/res/v1/web/search"
    params = {"q": query, "count": num_results}
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        results = response.json().get('web', {}).get('results', [])
        console.print(f"[info]Fetched {len(results)} results from Brave Search for query: {query}[/info]")
        return results
    except requests.RequestException as e:
        console.print(f"[warning]Brave Search API Error: {e}[/warning]")
        return []

# Scrape Webpage Function (new addition)
def scrape_webpage(url: str) -> str:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        console.print(f"[info]Scraped content from {url} (length: {len(text)})[/info]")
        return text[:10000]  # Limit to avoid token overflow
    except Exception as e:
        console.print(f"[warning]Error scraping {url}: {str(e)}[/warning]")
        return f"Error scraping {url}: {str(e)}"

# Define Agent class (needed for Web Search Mode)
class Agent:
    def __init__(self, client, system: str = "") -> None:
        self.client = client
        self.system = system
        self.messages: list = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message=""):
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        if provider == "Anthropic":
            response = self.client.messages.create(
                model=GENERAL_MODEL,
                max_tokens=8000,
                messages=self.messages,
                stream=True
            )
            full_response_content = ""
            if response.content:
                for part in response.content:
                    if hasattr(part, "text"):
                        full_response_content += part.text
            markdown_content = ""
            with Live(console=console, refresh_per_second=4) as live:
                for char in full_response_content:
                    markdown_content += char
                    live.update(Markdown(markdown_content))
                    time.sleep(0.02)
            print()
            return full_response_content
        elif provider == "Mistral":
            response = self.client.chat.stream(
                model=GENERAL_MODEL,
                messages=self.messages
            )
            full_response_content = ""
            markdown_content = ""
            with Live(console=console, refresh_per_second=4) as live:
                for chunk in response:
                    if chunk.data.choices[0].delta.content is not None:
                        full_response_content += chunk.data.choices[0].delta.content
                        markdown_content += chunk.data.choices[0].delta.content
                        live.update(Markdown(markdown_content))
            print()
            return full_response_content
        else:
            response = self.client.chat.completions.create(
                model=GENERAL_MODEL,
                messages=self.messages,
                stream=True
            )
            full_response_content = ""
            with Live(console=console, refresh_per_second=4) as live:
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunk_content = chunk.choices[0].delta.content
                        full_response_content += chunk_content
                        live.update(Markdown(full_response_content))
            print()
            return full_response_content

# Initialize agents after client is defined
def create_code_generator(client):
    return Agent(client=client, system="""
    You are a code generator. When a user asks for code, generate the code directly without any reasoning or observation.
    Provide the code in a markdown code block with the appropriate language syntax highlighting.
    """)

def create_web_app_generator(client):
    return Agent(client=client, system="""
    You are a web app generator. When a user asks for a web app, generate the code directly without any reasoning or observation.
    Provide the HTML and JavaScript code. Use Tailwind CSS classes for styling directly in the HTML. Do not include separate CSS code block, but use tailwind classes directly in the html.  Do not include tailwind script tag.
    Provide the HTML and JavaScript code in separate markdown code blocks with the appropriate language syntax highlighting.
    """)

def create_shell_agent(client):
    return Agent(client=client, system=f"""
    You are an advanced shell command assistant for {operating_system}.
    Your role is to:
    1. Generate safe, precise shell commands based on natural language requests
    2. Handle complex, multi-step tasks by breaking them into sequential commands
    3. Provide error handling suggestions when applicable
    4. Maintain awareness of the current directory: {os.getcwd()}
    
    Guidelines:
    - For simple tasks, return a single command
    - For complex tasks, return commands in a numbered list format (1., 2., 3., etc.)
    - Include safety checks (e.g., file existence) when relevant
    - Adjust syntax for {operating_system}:
      - Windows: use dir, cd, del, etc.
      - Linux/Termux/MacOS: use ls, cd, rm, etc.
    - Never use sudo/admin privileges
    - If a task is ambiguous, ask for clarification
    - If a task is potentially destructive, add a warning
    - Return commands only, no explanations unless clarification is needed
    
    Examples:
    - "list all text files": 
      Windows: "dir *.txt"
      Linux/Termux/MacOS: "ls *.txt"
    - "create a backup of my documents":
      1. dir "Documents" > nul 2>&1 || mkdir "Documents"
      2. xcopy "Documents" "Documents_Backup" /E /H /C /I
    """)

# Initialize global agents
code_generator = create_code_generator(client)
web_app_generator = create_web_app_generator(client)
shell_agent = create_shell_agent(client)  # Initialize shell_agent globally
code_writer = Agent(client=client, system="""
You are a code writer. When a user asks you to save code, you will save the code provided to a file with the filename given by the user.
""")

def change_model_provider():
    global client, provider, base_url, env_var_name, ROUTING_MODEL, TOOL_USE_MODEL, GENERAL_MODEL, IMAGE_MODEL, IMAGE_PROVIDER, image_api_key, code_generator, web_app_generator, shell_agent
    console.print(Panel(f"[bold]Current Provider: {provider}[/bold]", title="Current Provider"))
    new_provider, new_base_url, new_env_var_name = select_model()
    if new_provider is None:
        console.print("[info]Provider change cancelled.[/info]")
        return
    if input("Are you sure you want to change the provider? (yes/no): ").lower() != "yes":
        console.print("[info]Provider change cancelled.[/info]")
        return
    provider, base_url, env_var_name = new_provider, new_base_url, new_env_var_name
    if provider == "llama.cpp":
        try:
            response = requests.get("http://localhost:8080/v1/models", timeout=5)
            response.raise_for_status()
            new_model_name = response.json()["data"][0]["id"]
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error: Could not connect to local server. Error: {e}[/red]")
            new_model_name = "local model"
    elif provider == "Ollama":
        try:
            response = requests.get("http://localhost:11434/v1/models", timeout=5)
            response.raise_for_status()
            data = response.json()
            if "models" in data:
                console.print(Panel("[bold]Available Ollama Models[/bold]", title="Model Selection"))
                for idx, model in enumerate(data["models"], 1):
                    console.print(f"[bold white]{idx}.[/bold white] {model['name']}")
                while True:
                    model_choice = Prompt.ask("Select the number of the Ollama model")
                    try:
                        model_index = int(model_choice) - 1
                        if 0 <= model_index < len(data["models"]):
                            new_model_name = data["models"][model_index]["name"]
                            break
                        console.print("[warning]Invalid choice.[/warning]")
                    except ValueError:
                        console.print("[warning]Invalid choice. Enter a number.[/warning]")
            elif "data" in data and data["data"]:
                console.print(Panel("[bold]Available Ollama Models[/bold]", title="Model Selection"))
                for idx, model in enumerate(data["data"], 1):
                    console.print(f"[bold white]{idx}.[/bold white] {model['id']}")
                while True:
                    model_choice = Prompt.ask("Select the number of the Ollama model")
                    try:
                        model_index = int(model_choice) - 1
                        if 0 <= model_index < len(data["data"]):
                            new_model_name = data["data"][model_index]["id"]
                            break
                        console.print("[warning]Invalid choice.[/warning]")
                    except ValueError:
                        console.print("[warning]Invalid choice. Enter a number.[/warning]")
            else:
                console.print("[red]Error: No models found from Ollama server.[/red]")
                new_model_name = "llama3.2:latest"
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error: Could not connect to Ollama server. Error: {e}[/red]")
            new_model_name = "llama3.2:latest"
    else:
        console.print(Panel("[bold]Enter new model name[/bold]", title="Model Configuration"))
        new_model_name = Prompt.ask("Model name", default={
            "Groq": "llama-3.3-70b-versatile",
            "OpenRouter": "deepseek/deepseek-r1-zero:free",
            "Ollama": "local model",
            "llama.cpp": "local model",
            "Github": "gpt-4o",
            "Grok": "grok-2-latest",
            "Glhf": "hf:Qwen/Qwen2.5-Coder-32B-Instruct",
            "Anthropic": "claude-3-5-sonnet-20241022",
            "Openai": "gpt-4o",
            "Gemini": "gemini-2.0-flash-thinking-exp-01-21",
            "Mistral": "mistral-large-latest",
            "Huggingface": "meta-llama/Meta-Llama-3-8B-Instruct",
            "DeepSeek": "deepseek-chat",
            "Hyperbolic": "meta-llama/Llama-3.3-70B-Instruct",
            "Sambanova": "Meta-Llama-3.3-70B-Instruct",
            "Together.ai": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
        }.get(provider, "default-model"))
    if provider == "Glhf" and new_model_name != "hf:Qwen/Qwen2.5-Coder-32B-Instruct":
        new_model_name = "hf:" + new_model_name
    if new_provider == "Huggingface":
        new_api_key, new_base_url = get_api_key(new_provider, new_env_var_name)
    else:
        new_api_key = get_api_key(new_provider, new_env_var_name)
    os.environ[env_var_name] = new_api_key
    console.print(Panel("[bold]Select Image Generation Model Provider[/bold]", title="Image Model Provider"))
    image_provider_menu = """1. Huggingface"""
    console.print(Panel(image_provider_menu, title="[bold]Image Model Provider Menu[/bold]"))
    while True:
        image_provider_choice = Prompt.ask("Enter the number of your chosen image model provider", default="1")
        if image_provider_choice == "1":
            IMAGE_PROVIDER = "Huggingface"
            break
        console.print("[warning]Invalid choice. Please try again.[/warning]")
    console.print(Panel("[bold]Enter Image Generation model name[/bold]", title="Image Model Configuration"))
    new_image_model_name = Prompt.ask("Image Generation model name", default="black-forest-labs/FLUX.1-dev")
    if IMAGE_PROVIDER == "Huggingface":
        console.print(Panel(f"[bold]Enter Huggingface API Key (Leaving it blank sets Image Generation: off)[/bold]", title="Image Generation Model API Key Input"))
        image_api_key = get_masked_input(f"Enter Huggingface API Key (input will be masked): ")
        if not image_api_key.strip():
            console.print("[warning]API key not entered. Image Generation sets to off.[/warning]")
    else:
        image_api_key = None
    ROUTING_MODEL = TOOL_USE_MODEL = GENERAL_MODEL = new_model_name
    IMAGE_MODEL = new_image_model_name
    if provider == "Anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=new_api_key)
    elif provider == "Mistral":
        from mistralai import Mistral
        client = Mistral(api_key=new_api_key)
    elif provider == "Huggingface":
        from huggingface_hub import InferenceClient
        client = InferenceClient(base_url=new_base_url, api_key=new_api_key)
    else:
        client = openai.OpenAI(base_url=new_base_url, api_key=new_api_key)
    web_app_generator = create_web_app_generator(client)
    code_generator = create_code_generator(client)
    shell_agent = create_shell_agent(client)  # Reinitialize shell_agent
    console.print(Panel(f"[bold]Provider and Model Changed to: {provider} with model {new_model_name}[/bold]", title="Provider Changed"))
    return "Provider changed"

def run_file(filename):
    try:
        with open(filename, "r") as f:
            file_content = f.read()
    except FileNotFoundError:
        return f"File {filename} not found"
    except Exception as e:
        return f"Error reading file: {str(e)}"
    command = f"streamlit run {filename}" if 'import streamlit' in file_content else f"python3 {filename}"
    if not os.path.isfile("python3"):
        command = f"python {filename}"
    print(f"Command to run '{filename}': {command}")
    confirm = input("Do you want to execute this command? (yes/no): ")
    if confirm.lower() == "yes":
        try:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"Application running. Press (ctrl+c) and then (Enter) to stop...")
            try:
                while True:
                    output = process.stdout.readline()
                    if output:
                        print(output.strip())
                    if process.poll() is not None:
                        break
            except KeyboardInterrupt:
                print("\nStopping the application...")
                process.terminate()
                process.wait()
                return "Application stopped"
        except Exception as e:
            return f"Error executing the command: {str(e)}"
    return "Command not executed"

def help_command():
    help_text = Text.assemble(
        ("\n11KU7 AI CODER - Input Commands\n\n", "white"),
        ("op - Change provider & model\n", "green"),
        ("create a file <filename> - Create a new file\n", "green"),
        ("create a folder <foldername> - Create a new folder\n", "green"),
        ("list files - View current directory files\n", "green"),
        ("cd <directory> - Change current directory\n", "green"),
        ("code - activates code generation mode.\n", "green"),
        ("web app - activates web app generation mode.\n", "green"),
        ("ask code - activates ask code mode.\n", "green"),
        ("ask image - activates ask image mode.\n", "green"),
        ("ask url - activates ask url mode.\n", "green"),
        ("ask docu - activates ask document mode.\n", "green"),
        ("image - activates image generation mode.\n", "green"),
        ("shell agent - activates shell agent mode.\n", "green"),
        ("vitex app - activates vite app mode.\n", "green"),
        ("gitrepo - activates Git Repository Mode.\n", "green"),
        ("web search - activates Web Search Mode.\n", "green"),
        ("shell - Execute shell commands\n", "green"),
        ("run file <filename> - run a python file\n", "green"),
        ("quit - Exit the program\n", "green"),
        ("clear screen - Clear the console\n", "green"),
        ("show menu - Display the menu\n", "green"),
        ("help - help menu\n", "green"),
        ("install help - optional dependencies installation help\n", "green"),
        ("install poppler - Install Poppler for PDF-to-image features (Windows only)\n", "green"),
        ("prompt - change system prompt\n", "green"),
    )
    console.print(Panel(help_text, title="[bold]Help Menu", border_style="white"))

def install_help_command():
    help_text = Text.assemble(
        ("\n11KU7 AI CODER - Optional external dependencies installation instructions\n\n", "white"),
        (
"""
Dependency name = poppler
This is an optional dependency, without it [ask docu] mode can be queried about text present in pdf document.
This installation is needed in [ask docu] mode to query about images present in pdf document.

**Installation Instruction-**
[Linux/Termux]
pkg install poppler

**Installation Instruction-**
[Windows]
1) Download poppler latest release from github
https://github.com/oschwartz10612/poppler-windows/releases/download/v24.08.0-0/Release-24.08.0-0.zip
2) Extract this file to any directory for example-
C:\\poppler\\
3) Go to bin folder, and then copy the path like-
C:\\poppler\\poppler-24.08.0\\Library\\bin
4) Add above path to "path" system variable by clicking "new" and pasting above path and then OK to apply
5) Restart the command prompt

Verify Installation:
Open a new command prompt and type-
pdftoppm -v
if showing poppler version number, it is correctly installed and configured.\n""", "green"),
    )
    console.print(Panel(help_text, title="[bold]Install Help", border_style="white"))

def execute_shell_command():
    console.print(Panel("Entered Shell Command Mode", style="bold green"))
    
    def get_operating_system():
        """Detect the operating system for platform-specific commands."""
        if os.name == "nt":
            return "Windows"
        elif "TERMUX_VERSION" in os.environ:
            return "Termux"
        elif sys.platform == "darwin":
            return "MacOS"
        elif sys.platform.startswith("linux"):
            return "Linux"
        return "Unknown"

    os_name = get_operating_system()
    shell_agent = Agent(client=client, system=f"""
    You are a shell command generator and executor for Shell Command Mode.
    The current working directory is: {os.getcwd()}.
    The operating system is: {os_name}.
    Based on the user's query, generate shell commands relevant to the current directory.
    Provide commands as plain text without backticks or language identifiers.
    - For Windows, use CMD commands (e.g., 'dir /b *.txt') combining extensions into a single line when listing multiple file types.
    - For Linux, MacOS, or Termux, use Unix commands (e.g., 'find . -name "*.txt"').
    For multi-line scripts, provide the full script as a single block of text with proper separators (e.g., '&&' for Windows, ';' for Unix).
    Do not execute commands; only generate them for user approval.
    If the query is unclear, ask for clarification.
    """)

    def clean_shell_commands(commands: str) -> str:
        """Clean shell commands by removing Markdown markers."""
        commands = re.sub(r'```(?:sh|bash|cmd)?\s*', '', commands, flags=re.MULTILINE)
        commands = commands.replace('```', '').strip()
        return commands

    def execute_shell_script(script: str) -> bool:
        """Execute a shell script or command based on OS."""
        if os_name == "Windows":
            temp_script = os.path.join(os.getcwd(), "temp_script.bat")
            try:
                lines = script.splitlines()
                if all(line.strip().startswith("dir /b") for line in lines if line.strip()):
                    combined = " ".join(line.strip() for line in lines if line.strip())
                    script = combined
                with open(temp_script, 'w', encoding='utf-8') as f:
                    f.write(f"@echo off\ncd /d \"{os.getcwd()}\"\n{script}")
                cmd = f"cmd.exe /c \"{temp_script}\""
                console.print(Panel(f"Executing shell script:\n{script}", style="cyan"))
                consent = Prompt.ask("Execute this script? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                if consent == "y":
                    try:
                        if script.strip().split()[0].lower() in ['python', 'java', 'node', 'ruby', 'perl', 'php', 'go', 'rust', 'gcc', 'g++', 'run', 'execute']:
                            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                            console.print("Command is running. Press (Ctrl+C) and then (Enter) to stop...")
                            try:
                                while True:
                                    output = process.stdout.readline()
                                    if output:
                                        console.print(output.strip())
                                    if process.poll() is not None:
                                        break
                            except KeyboardInterrupt:
                                console.print("\nStopping the command execution...")
                                process.terminate()
                                process.wait()
                                return False
                        else:
                            result = subprocess.check_output(cmd, shell=True, text=True).strip()
                            console.print(Panel(f"Command output:\n{result.replace('\n', '\n│ ')}", style="green"))
                            return True
                    except subprocess.CalledProcessError as e:
                        console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                        return False
                return False
            finally:
                if os.path.exists(temp_script):
                    os.remove(temp_script)
        else:  # Unix-like
            temp_script = os.path.join(os.getcwd(), "temp_script.sh")
            try:
                with open(temp_script, 'w', encoding='utf-8') as f:
                    f.write(f"#!/bin/bash\ncd \"{os.getcwd()}\"\n{script}")
                os.chmod(temp_script, 0o755)
                cmd = f"bash \"{temp_script}\""
                console.print(Panel(f"Executing shell script:\n{script}", style="cyan"))
                consent = Prompt.ask("Execute this script? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                if consent == "y":
                    try:
                        if script.strip().split()[0].lower() in ['python', 'java', 'node', 'ruby', 'perl', 'php', 'go', 'rust', 'gcc', 'g++', 'run', 'execute']:
                            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                            console.print("Command is running. Press (Ctrl+C) and then (Enter) to stop...")
                            try:
                                while True:
                                    output = process.stdout.readline()
                                    if output:
                                        console.print(output.strip())
                                    if process.poll() is not None:
                                        break
                            except KeyboardInterrupt:
                                console.print("\nStopping the command execution...")
                                process.terminate()
                                process.wait()
                                return False
                        else:
                            result = subprocess.check_output(cmd, shell=True, text=True).strip()
                            console.print(Panel(f"Command output:\n{result.replace('\n', '\n│ ')}", style="green"))
                            return True
                    except subprocess.CalledProcessError as e:
                        console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                        return False
                return False
            finally:
                if os.path.exists(temp_script):
                    os.remove(temp_script)

    while True:
        choice = input("Do you want to [1] enter a command directly, [2] describe what you want to do, or [q]uit? (1/2/q): ").strip().lower()
        if choice == "q":
            console.print(Panel("Exited Shell Command Mode", style="bold green"))
            break
        elif choice not in ["1", "2"]:
            console.print("[red]Invalid choice. Please enter '1', '2', or 'q'.[/red]")
            continue

        if choice == "1":
            user_input = input("Enter the command to execute: ").strip()
            command = user_input
        else:
            user_input = input("Describe what you want to do (e.g., 'list files'): ").strip()
            try:
                command = clean_shell_commands(shell_agent(user_input))
                # Verification step from original code
                verification_prompt = f"""
                Please verify if the command '{command}' matches the user's request to '{user_input}'
                on the operating system '{os_name}'.
                **Respond with 'Yes' if it matches, or 'No' if it does not match.**"""
                verification_response = client.chat.completions.create(
                    model=GENERAL_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a command verifier for shell operations."},
                        {"role": "user", "content": verification_prompt},
                    ],
                )
                if verification_response.choices[0].message.content.strip().lower() == "no":
                    console.print(f"[yellow]Warning: The suggested command '{command}' might not match your request.[/yellow]")
            except Exception as e:
                console.print(Panel(f"Error generating command: {e}", style="red"))
                continue

        console.print(Panel(command, title="Generated Shell Command", style="cyan"))
        command_lines = command.splitlines()

        exec_choice = Prompt.ask("Execute command? ([a]ll, [s]tep-by-step, [n]o)", choices=["a", "s", "n"], default="n", console=console)
        if exec_choice == "a":
            if len(command_lines) > 1 or any(line.strip().startswith("for") or line.strip().startswith("#!") for line in command_lines):
                if not execute_shell_script(command):
                    console.print(Panel("Script execution failed or skipped.", style="yellow"))
            else:
                for cmd in command_lines:
                    if cmd.strip():
                        console.print(Panel(f"Executing shell command:\n{cmd}", style="cyan"))
                        consent = Prompt.ask("Execute this command? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                        if consent == "y":
                            try:
                                if cmd.strip().split()[0].lower() in ['python', 'java', 'node', 'ruby', 'perl', 'php', 'go', 'rust', 'gcc', 'g++', 'run', 'execute']:
                                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                                    console.print("Command is running. Press (Ctrl+C) and then (Enter) to stop...")
                                    try:
                                        while True:
                                            output = process.stdout.readline()
                                            if output:
                                                console.print(output.strip())
                                            if process.poll() is not None:
                                                break
                                    except KeyboardInterrupt:
                                        console.print("\nStopping the command execution...")
                                        process.terminate()
                                        process.wait()
                                        break
                                else:
                                    result = subprocess.check_output(cmd, shell=True, text=True).strip()
                                    console.print(Panel(f"Command output:\n{result.replace('\n', '\n│ ')}", style="green"))
                            except subprocess.CalledProcessError as e:
                                console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                                console.print(Panel("Command execution failed or skipped.", style="yellow"))
                                break
        elif exec_choice == "s":
            for cmd in command_lines:
                if cmd.strip():
                    console.print(Panel(f"Next command: {cmd}", style="cyan"))
                    consent = Prompt.ask("Execute this command? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                    if consent == "y":
                        try:
                            if cmd.strip().split()[0].lower() in ['python', 'java', 'node', 'ruby', 'perl', 'php', 'go', 'rust', 'gcc', 'g++', 'run', 'execute']:
                                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                                console.print("Command is running. Press (Ctrl+C) and then (Enter) to stop...")
                                try:
                                    while True:
                                        output = process.stdout.readline()
                                        if output:
                                            console.print(output.strip())
                                        if process.poll() is not None:
                                            break
                                except KeyboardInterrupt:
                                    console.print("\nStopping the command execution...")
                                    process.terminate()
                                    process.wait()
                                    break
                            else:
                                result = subprocess.check_output(cmd, shell=True, text=True).strip()
                                console.print(Panel(f"Command output:\n{result.replace('\n', '\n│ ')}", style="green"))
                        except subprocess.CalledProcessError as e:
                            console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                            console.print(Panel("Command execution stopped.", style="yellow"))
                            break

def change_directory(new_directory):
    try:
        os.chdir(new_directory)
        console.print(Panel(f"[success]Directory changed to: {os.getcwd()}[/success]", title="Directory Change"))
    except FileNotFoundError:
        console.print(Panel(f"[warning]Directory not found: {new_directory}[/warning]", title="Directory Change"))
    except Exception as e:
        console.print(Panel(f"[warning]Error changing directory: {e}[/warning]", title="Directory Change"))

def save_conversation(conversation, filename, provider, model_name, start_time):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("### 11KU7-ai-coder (version: 1.0)\n\n")
            f.write(f"### Model Provider:\n\n   {provider}\n   {model_name}\n\n")
            formatted_start_time = start_time.strftime("%Y-%m-%d %H:%M:%S") + " UTC"
            f.write(f"### Program Started on:\n\n   {formatted_start_time}\n\n")
            for message in conversation:
                role = message["role"]
                content = message["content"]
                f.write(f"### {role.capitalize()}:\n\n{content}\n\n")
        console.print(Panel(f"[bold]Conversation saved to {filename}[/bold]", title="Success", border_style="green"))
        return True
    except Exception as e:
        console.print(Panel(f"[bold]Error saving conversation: {str(e)}[/bold]", title="Error", border_style="red"))
        return False

from rich.live import Live
from rich.markdown import Markdown

class Agent:
    def __init__(self, client, system: str = "") -> None:
        self.client = client
        self.system = system
        self.messages: list = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message=""):
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        if provider == "Anthropic":
            response = self.client.messages.create(
                model=GENERAL_MODEL,
                max_tokens=8000,
                messages=self.messages,
                stream=True
            )
            full_response_content = ""
            if response.content:
                for part in response.content:
                    if hasattr(part, "text"):
                        full_response_content += part.text
            markdown_content = ""
            with Live(console=console, refresh_per_second=4) as live:
                for char in full_response_content:
                    markdown_content += char
                    live.update(Markdown(markdown_content))
                    time.sleep(0.02)
            print()
            return full_response_content
        elif provider == "Mistral":
            response = self.client.chat.stream(
                model=GENERAL_MODEL,
                messages=self.messages
            )
            full_response_content = ""
            markdown_content = ""
            with Live(console=console, refresh_per_second=4) as live:
                for chunk in response:
                    if chunk.data.choices[0].delta.content is not None:
                        full_response_content += chunk.data.choices[0].delta.content
                        markdown_content += chunk.data.choices[0].delta.content
                        live.update(Markdown(markdown_content))
            print()
            return full_response_content
        else:
            response = self.client.chat.completions.create(
                model=GENERAL_MODEL,
                messages=self.messages,
                stream=True
            )
            full_response_content = ""
            with Live(console=console, refresh_per_second=4) as live:
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunk_content = chunk.choices[0].delta.content
                        full_response_content += chunk_content
                        live.update(Markdown(full_response_content))
            print()
            return full_response_content

# Enhanced Shell Agent
def create_shell_agent(client):
    return Agent(client=client, system=f"""
    You are an advanced shell command assistant for {operating_system}.
    Your role is to:
    1. Generate safe, precise shell commands based on natural language requests
    2. Handle complex, multi-step tasks by breaking them into sequential commands
    3. Provide error handling suggestions when applicable
    4. Maintain awareness of the current directory: {os.getcwd()}
    
    Guidelines:
    - For simple tasks, return a single command
    - For complex tasks, return commands in a numbered list format (1., 2., 3., etc.)
    - Include safety checks (e.g., file existence) when relevant
    - Adjust syntax for {operating_system}:
      - Windows: use dir, cd, del, etc.
      - Linux/Termux/MacOS: use ls, cd, rm, etc.
    - Never use sudo/admin privileges
    - If a task is ambiguous, ask for clarification
    - If a task is potentially destructive, add a warning
    - Return commands only, no explanations unless clarification is needed
    
    Examples:
    - "list all text files": 
      Windows: "dir *.txt"
      Linux/Termux/MacOS: "ls *.txt"
    - "create a backup of my documents":
      1. dir "Documents" > nul 2>&1 || mkdir "Documents"
      2. xcopy "Documents" "Documents_Backup" /E /H /C /I
    """)

def shell_agent_mode():
    global shell_agent, provider, ROUTING_MODEL, start_time
    conversation = []
    command_history = []  # Store executed commands
    
    while True:
        console.print(Panel("**Shell Agent Mode**", title="Shell Agent Mode", style="bold red"))
        console.print(Panel(f"Current directory: {os.getcwd()}", style="green"))
        
        # Show recent command history
        if command_history:
            console.print(Panel(
                "\n".join(f"{i+1}. {cmd}" for i, cmd in enumerate(command_history[-3:])),
                title="Recent Commands (Last 3)",
                style="green"
            ))
            
        query_prompt = ("Enter your shell task in natural language\n"
                       "(or 'quit' to exit, 'save' to save conversation, "
                       "'history' to see full history): ")
        query = Prompt.ask(Text(query_prompt, style="green"))
        conversation.append({"role": "user", "content": query_prompt + query})

        # Handle special commands
        if query.lower() == 'quit':
            console.print(Panel("**Exited Shell Agent Mode**", style="bold red"))
            break
            
        elif query.lower() == 'save':
            if conversation:
                filename = Prompt.ask("Enter filename (e.g., shell_session.md) or Enter to skip")
                if filename and save_conversation(conversation, filename, provider, ROUTING_MODEL, start_time):
                    if operating_system == "Termux":
                        handle_termux_file_options(filename)
            continue
            
        elif query.lower() == 'history':
            if command_history:
                console.print(Panel(
                    "\n".join(f"{i+1}. {cmd}" for i, cmd in enumerate(command_history)),
                    title="Command History",
                    style="bold green"
                ))
            else:
                console.print(Panel("No commands in history yet", style="bold yellow"))
            continue

        try:
            # Generate command(s) using the shell agent
            commands = shell_agent(query)
            conversation.append({"role": "assistant", "content": commands})
            
            # Parse commands (could be single line or numbered list)
            command_list = []
            if commands.strip():
                if "\n" in commands:
                    # Multi-line commands with numbering
                    for line in commands.split("\n"):
                        line = line.strip()
                        if line and re.match(r"^\d+\.\s", line):
                            command_list.append(line.split(".", 1)[1].strip())
                        elif line:  # Handle unnumbered lines (warnings/clarifications)
                            console.print(Panel(line, style="bold yellow"))
                else:
                    command_list = [commands.strip()]
            
            if not command_list:
                console.print(Panel("No valid commands generated", style="bold yellow"))
                continue
                
            # Display generated commands
            console.print(Panel(
                "\n".join(f"{i+1}. {cmd}" for i, cmd in enumerate(command_list)),
                title="Generated Commands",
                style="green"
            ))
            
            # Execution options
            exec_choice = Prompt.ask(
                "Execute commands? ([a]ll, [s]tep-by-step, [n]o)",
                choices=["a", "s", "n"], default="n"
            )
            
            if exec_choice == "n":
                console.print(Panel("Execution cancelled", style="bold yellow"))
                continue
                
            # Execute commands
            results = []
            for i, cmd in enumerate(command_list):
                if exec_choice == "s":
                    confirm = Prompt.ask(f"Execute '{cmd}'? (y/n)", choices=["y", "n"], default="n")
                    if confirm != "y":
                        results.append(f"Command {i+1} skipped")
                        continue
                
                try:
                    # Add basic safety check
                    if any(danger in cmd.lower() for danger in ["rm -rf", "del /f", "format", "drop"]):
                        console.print(Panel(
                            f"Potentially dangerous command detected: {cmd}",
                            style="bold red"
                        ))
                        if Prompt.ask("Proceed anyway? (y/n)", default="n") != "y":
                            results.append(f"Command {i+1} skipped (safety check)")
                            continue
                    
                    result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                    results.append(f"Command {i+1} output:\n{result}")
                    command_history.append(cmd)
                    
                except subprocess.CalledProcessError as e:
                    results.append(f"Command {i+1} failed:\n{e.output}")
                except Exception as e:
                    results.append(f"Command {i+1} error: {str(e)}")
                
                if exec_choice == "s" and i < len(command_list) - 1:
                    console.print(Panel(results[-1], style="bold green" if "failed" not in results[-1] else "bold red"))
            
            # Show all results
            if results:
                console.print(Panel(
                    "\n".join(results),
                    title="Execution Results",
                    style="bold green" if all("failed" not in r for r in results) else "bold red"
                ))
                
        except Exception as e:
            console.print(Panel(f"Error: {str(e)}", title="Error", style="bold red"))

def create_code_generator(client):
    return Agent(client=client, system="""
    You are a code generator. When a user asks for code, generate the code directly without any reasoning or observation.
    Provide the code in a markdown code block with the appropriate language syntax highlighting.
    """)
code_generator = create_code_generator(client)

code_writer = Agent(client=client, system="""
You are a code writer. When a user asks you to save code, you will save the code provided to a file with the filename given by the user.
""")

def code():
    global code_generator, provider, ROUTING_MODEL, start_time
    conversation = []
    while True:
        console.print(Panel("**Code Generation Mode**", title="Code Generation Mode", style="bold red"))
        query_prompt = f"Enter your code query (or 'quit' to exit code mode or 'save' to save conversation): "
        query = Prompt.ask(Text(query_prompt, style="green"))
        conversation.append({"role": "user", "content": query_prompt + query})
        if query.lower() == 'quit':
            console.print(Panel("**Exited Code Generation Mode**", style="bold red"))
            break
        if query.lower() == "save":
            if conversation:
                filename = Prompt.ask("Enter the filename to save the conversation to (e.g., my_conversation.md) or press Enter to skip")
                if filename and save_conversation(conversation, filename, provider, ROUTING_MODEL, start_time):
                    if operating_system == "Termux":
                        handle_termux_file_options(filename)
            continue
        try:
            full_response_content = code_generator(query)
            conversation.append({"role": "assistant", "content": full_response_content})
            match = re.search(r'```(\w+)?\n(.*?)```', full_response_content, re.DOTALL)
            if match:
                language = match.group(1) or "python"
                code = match.group(2)
                console.print(Panel(Markdown(f"```{language}\n{code}```"), title="[bold]Code preview", border_style="green"))
                filename = Prompt.ask("Enter the filename to save the code to (or press Enter to skip)")
                if filename:
                    try:
                        with open(filename, 'w', encoding='utf-8') as file:
                            file.write(code)
                        console.print(Panel(f"Code has been written to [success]{filename}[/success]", title="File Saved"))
                        if operating_system == "Termux":
                            handle_termux_file_options(filename)
                    except Exception as e:
                        console.print(Markdown(f"**Error saving file: {e}**"))
            else:
                console.print(Panel(Markdown(full_response_content), title="[bold]Generated Response", border_style="green"))
        except Exception as e:
            console.print(Panel(f"Error: {str(e)}, try again.", title="Error", style="bold red"))

def encode_image(image_path):
    try:
        image_path = image_path.strip('"')
        if os.path.isfile(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        return image_path
    except FileNotFoundError:
        console.print(Panel(f"Error: File '{image_path}' not found.", title="File Not Found", style="bold red"))
        return None
    except OSError as e:
        console.print(Panel(f"Error: {e}. Please check the file path.", title="Invalid File Path", style="bold red"))
        return None

def ask_the_image_mode():
    global provider, ROUTING_MODEL, start_time
    console.print(Panel("**Ask the Image Mode Activated**", style="bold red"))
    conversation = []

    def get_operating_system():
        """Detect the operating system for platform-specific commands."""
        if os.name == "nt":
            return "Windows"
        elif "TERMUX_VERSION" in os.environ:
            return "Termux"
        elif sys.platform == "darwin":
            return "MacOS"
        elif sys.platform.startswith("linux"):
            return "Linux"
        return "Unknown"

    def shell_agent_mode(conversation: list, image_path: str = "", base64_image: str = ""):
        """Shell agent submode for generating and executing shell commands."""
        os_name = get_operating_system()
        shell_agent = Agent(client=client, system=f"""
        You are a shell command generator and executor for Ask Image Mode.
        The current working directory is: {os.getcwd()}.
        The operating system is: {os_name}.
        {'The current image path is: ' + image_path if image_path else 'No image loaded yet.'}
        {'An image is loaded.' if base64_image else 'No image loaded yet.'}
        Based on the user's query, generate shell commands relevant to the current directory or image context.
        Provide commands as plain text without backticks or language identifiers.
        - For Windows, use CMD commands (e.g., 'dir /b *.jpg *.jpeg *.png') combining extensions into a single line when listing multiple file types.
        - For Linux, MacOS, or Termux, use Unix commands (e.g., 'find . -name "*.jpg" -o -name "*.png"').
        For multi-line scripts, provide the full script as a single block of text with proper separators (e.g., '&&' for Windows, ';' for Unix).
        Do not execute commands; only generate them for user approval.
        If the query is unclear, ask for clarification.
        """)
        console.print(Panel(f"Entered Shell Agent Mode (OS: {os_name}). Request shell commands or type 'quit-shell' to exit.", style="bold green"))

        def clean_shell_commands(commands: str) -> str:
            """Clean shell commands by removing Markdown markers."""
            commands = re.sub(r'```(?:sh|bash|cmd)?\s*', '', commands, flags=re.MULTILINE)
            commands = commands.replace('```', '').strip()
            return commands

        def execute_shell_script(script: str) -> bool:
            """Execute a shell script or command based on OS."""
            if os_name == "Windows":
                temp_script = os.path.join(os.getcwd(), "temp_script.bat")
                try:
                    # Combine dir commands into one line for Windows if multiple dir /b lines exist
                    lines = script.splitlines()
                    if all(line.strip().startswith("dir /b") for line in lines if line.strip()):
                        combined = " ".join(line.strip() for line in lines if line.strip())
                        script = combined
                    with open(temp_script, 'w', encoding='utf-8') as f:
                        f.write(f"@echo off\ncd /d \"{os.getcwd()}\"\n{script}")
                    cmd = f"cmd.exe /c \"{temp_script}\""
                    console.print(Panel(f"Executing shell script:\n{script}", style="cyan"))
                    consent = Prompt.ask("Execute this script? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                    if consent == "y":
                        try:
                            result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                            console.print(Panel(f"Command output:\n{result}", style="green"))
                            return True
                        except subprocess.CalledProcessError as e:
                            console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                            return False
                    return False
                finally:
                    if os.path.exists(temp_script):
                        os.remove(temp_script)
            else:  # Unix-like
                temp_script = os.path.join(os.getcwd(), "temp_script.sh")
                try:
                    with open(temp_script, 'w', encoding='utf-8') as f:
                        f.write(f"#!/bin/bash\ncd \"{os.getcwd()}\"\n{script}")
                    os.chmod(temp_script, 0o755)
                    cmd = f"bash \"{temp_script}\""
                    console.print(Panel(f"Executing shell script:\n{script}", style="cyan"))
                    consent = Prompt.ask("Execute this script? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                    if consent == "y":
                        try:
                            result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                            console.print(Panel(f"Command output:\n{result}", style="green"))
                            return True
                        except subprocess.CalledProcessError as e:
                            console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                            return False
                    return False
                finally:
                    if os.path.exists(temp_script):
                        os.remove(temp_script)

        while True:
            shell_query = input("Shell query (or 'quit-shell' to exit): ").strip()
            if shell_query.lower() == "quit-shell":
                console.print(Panel("Exited Shell Agent Mode", style="bold green"))
                break
            conversation.append({"role": "user", "content": shell_query})
            try:
                shell_commands = shell_agent(shell_query)
                cleaned_commands = clean_shell_commands(shell_commands)
                console.print(Panel(cleaned_commands, title="Generated Shell Commands", style="cyan"))
                conversation.append({"role": "assistant", "content": cleaned_commands})

                exec_choice = Prompt.ask("Execute commands? ([a]ll, [s]tep-by-step, [n]o)", choices=["a", "s", "n"], default="n", console=console)
                command_lines = cleaned_commands.splitlines()

                if exec_choice == "a":
                    if len(command_lines) > 1 or any(line.strip().startswith("for") or line.strip().startswith("#!") for line in command_lines):
                        if not execute_shell_script(cleaned_commands):
                            console.print(Panel("Script execution failed or skipped.", style="yellow"))
                    else:
                        for cmd in command_lines:
                            if cmd.strip():
                                console.print(Panel(f"Executing shell command:\n{cmd}", style="cyan"))
                                consent = Prompt.ask("Execute this command? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                                if consent == "y":
                                    try:
                                        result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                                        console.print(Panel(f"Command output:\n{result}", style="green"))
                                    except subprocess.CalledProcessError as e:
                                        console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                                        console.print(Panel("Command execution failed or skipped.", style="yellow"))
                                        break
                elif exec_choice == "s":
                    for cmd in command_lines:
                        if cmd.strip():
                            console.print(Panel(f"Next command: {cmd}", style="cyan"))
                            consent = Prompt.ask("Execute this command? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                            if consent == "y":
                                try:
                                    result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                                    console.print(Panel(f"Command output:\n{result}", style="green"))
                                except subprocess.CalledProcessError as e:
                                    console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                                    console.print(Panel("Command execution stopped.", style="yellow"))
                                    break
            except Exception as e:
                console.print(Panel(f"Error generating or executing shell commands: {e}", style="red"))

    while True:
        image_path_prompt = f"Enter the path to your image file or URL (or 'quit' to exit, 'save' to save conversation, 'shell' for shell commands): "
        image_path = Prompt.ask(image_path_prompt)
        conversation.append({"role": "user", "content": image_path_prompt + image_path})

        if image_path.lower() == 'quit':
            console.print(Panel("**Exited Ask the Image Mode**", style="bold red"))
            break
        elif image_path.lower() == "save":
            if conversation:
                filename = Prompt.ask("Enter the filename to save the conversation to (e.g., my_conversation.md) or press Enter to skip")
                if filename and save_conversation(conversation, filename, provider, ROUTING_MODEL, start_time):
                    if operating_system == "Termux":
                        handle_termux_file_options(filename)
            continue
        elif image_path.lower() == "shell":
            shell_agent_mode(conversation)
            continue

        base64_image = encode_image(image_path)
        if base64_image is None:
            continue
        conversation.append({"role": "user", "content": f"Image Path: {image_path}"})
        previous_responses = []
        query_prompt = "What would you like to know about the image?\n\n"
        console.print(query_prompt)

        while True:
            user_query_prompt = f"Enter your query (Type 'new' to query a new image, 'quit' to exit, 'save' to save conversation, 'shell' for shell commands): "
            user_query = Prompt.ask(Text(user_query_prompt, style="green"))
            conversation.append({"role": "user", "content": user_query_prompt + user_query})

            if user_query.lower() == 'new':
                console.print("[bold]Starting new image query...[/bold]")
                break
            elif user_query.lower() == 'quit':
                console.print(Panel("**Exited Ask the Image Mode**", style="bold red"))
                return
            elif user_query.lower() == "save":
                if conversation:
                    filename = Prompt.ask("Enter the filename to save the conversation to (e.g., my_conversation.md) or press Enter to skip")
                    if filename and save_conversation(conversation, filename, provider, ROUTING_MODEL, start_time):
                        if operating_system == "Termux":
                            handle_termux_file_options(filename)
                continue
            elif user_query.lower() == "shell":
                shell_agent_mode(conversation, image_path, base64_image)
                continue

            full_query = "\n".join(previous_responses) + "\n" + user_query
            conversation.append({"role": "user", "content": full_query})
            try:
                if provider == "Anthropic":
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_query},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}}
                        ],
                    }]
                    response = client.messages.create(model=GENERAL_MODEL, max_tokens=8000, messages=messages, stream=True)
                    full_response_content = ""
                    if response.content:
                        for part in response.content:
                            if hasattr(part, "text"):
                                full_response_content += part.text
                    markdown_content = ""
                    with Live(console=console, refresh_per_second=4) as live:
                        for char in full_response_content:
                            markdown_content += char
                            live.update(Markdown(markdown_content))
                            time.sleep(0.02)
                    print()
                elif provider == "Mistral":
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_query},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}" if not base64_image.startswith("http") else f"{base64_image}"}},
                        ],
                    }]
                    response = client.chat.stream(model=GENERAL_MODEL, messages=messages)
                    full_response_content = ""
                    markdown_content = ""
                    with Live(console=console, refresh_per_second=4) as live:
                        for chunk in response:
                            if chunk.data.choices[0].delta.content is not None:
                                chunk_content = chunk.data.choices[0].delta.content
                                full_response_content += chunk_content
                                markdown_content += chunk_content
                                live.update(Markdown(markdown_content))
                    print()
                else:
                    response = client.chat.completions.create(
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": full_query},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}" if not base64_image.startswith("http") else f"{base64_image}"}},
                            ],
                        }],
                        model=GENERAL_MODEL,
                        stream=True
                    )
                    full_response_content = ""
                    markdown_content = ""
                    with Live(console=console, refresh_per_second=4) as live:
                        for chunk in response:
                            if chunk.choices and chunk.choices[0].delta.content:
                                chunk_content = chunk.choices[0].delta.content
                                full_response_content += chunk_content
                                markdown_content += chunk_content
                                live.update(Markdown(markdown_content))
                    print()

                match = re.search(r'```(\w+)?\n(.*?)```', full_response_content, re.DOTALL)
                if match:
                    language = match.group(1) or "python"
                    code = match.group(2)
                    console.print(Markdown(f"```{language}\n{code}```"))
                    filename = Prompt.ask("Enter the filename to save the code to (or press Enter to skip)")
                    if filename:
                        with open(filename, 'w', encoding='utf-8') as file:
                            file.write(code)
                        console.print(Panel(f"Code has been written to [success]{filename}[/success]", title="File Saved"))
                response = full_response_content
                console.print(Markdown(response))
                previous_responses.append(response)
                conversation.append({"role": "assistant", "content": response})
            except Exception as e:
                console.print(Panel(f"Error: {str(e)}, try again.", title="Error", style="bold red"))



def validate_url(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?|localhost|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::\d+)?(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

def fetch_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        console.print(Panel(f"Failed to fetch URL content: {str(e)}", title="Error", style="bold red"))
        return None

def ask_url_mode():
    global provider, ROUTING_MODEL, start_time
    console.print(Panel("**Ask URL Mode Activated**", style="bold red"))
    conversation = []
    while True:
        url_prompt = f"Enter the URL to query (or 'quit' to exit or 'save' to save conversation): "
        url = Prompt.ask(url_prompt)
        conversation.append({"role": "user", "content": url_prompt + url})
        if url.lower() == 'quit':
            console.print(Panel("**Exited Ask URL Mode**", style="bold red"))
            break
        if url.lower() == "save":
            if conversation:
                filename = Prompt.ask("Enter the filename to save the conversation to (e.g., my_conversation.md) or press Enter to skip")
                if filename and save_conversation(conversation, filename, provider, ROUTING_MODEL, start_time):
                    if operating_system == "Termux":
                        handle_termux_file_options(filename)
            continue
        if not validate_url(url):
            console.print(Panel("Invalid URL. Please enter a valid URL.", style="bold red"))
            continue
        url_content = fetch_url_content(url)
        if url_content is None:
            continue
        previous_responses = []
        query_prompt = "What would you like to know about the URL?\n\n"
        console.print(query_prompt)
        while True:
            user_query_prompt = f"Enter your query (Type 'new' to query a new URL or 'quit' to exit or 'save' to save conversation): "
            user_query = Prompt.ask(Text(user_query_prompt, style="green"))
            conversation.append({"role": "user", "content": user_query_prompt + user_query})
            if user_query.lower() == 'new':
                console.print("[bold]Starting new URL query...[/bold]")
                break
            elif user_query.lower() == 'quit':
                console.print(Panel("**Exited Ask URL Mode**", style="bold red"))
                return
            elif user_query.lower() == "save":
                if conversation:
                    filename = Prompt.ask("Enter the filename to save the conversation to (e.g., my_conversation.md) or press Enter to skip")
                    if filename and save_conversation(conversation, filename, provider, ROUTING_MODEL, start_time):
                        if operating_system == "Termux":
                            handle_termux_file_options(filename)
                continue
            full_query = "\n".join(previous_responses) + "\n" + user_query
            try:
                if provider == "Anthropic":
                    messages = [{"role": "user", "content": [{"type": "text", "text": full_query}, {"type": "text", "text": f"URL Content: {url_content}"}]}]
                    response = client.messages.create(model=GENERAL_MODEL, max_tokens=8000, messages=messages, stream=True)
                    full_response_content = ""
                    if response.content:
                        for part in response.content:
                            if hasattr(part, "text"):
                                full_response_content += part.text
                    markdown_content = ""
                    with Live(console=console, refresh_per_second=4) as live:
                        for char in full_response_content:
                            markdown_content += char
                            live.update(Markdown(markdown_content))
                            time.sleep(0.02)
                    print()
                elif provider == "Mistral":
                    messages = [{"role": "user", "content": [{"type": "text", "text": full_query}, {"type": "text", "text": f"URL Content: {url_content}"}]}]
                    response = client.chat.stream(model=GENERAL_MODEL, messages=messages)
                    full_response_content = ""
                    markdown_content = ""
                    with Live(console=console, refresh_per_second=4) as live:
                        for chunk in response:
                            if chunk.data.choices[0].delta.content is not None:
                                chunk_content = chunk.data.choices[0].delta.content
                                full_response_content += chunk_content
                                markdown_content += chunk_content
                                live.update(Markdown(markdown_content))
                    print()
                else:
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": [{"type": "text", "text": full_query}, {"type": "text", "text": f"URL Content: {url_content}"}]}],
                        model=GENERAL_MODEL,
                        stream=True
                    )
                    full_response_content = ""
                    markdown_content = ""
                    with Live(console=console, refresh_per_second=4) as live:
                        for chunk in response:
                            if chunk.choices and chunk.choices[0].delta.content:
                                chunk_content = chunk.choices[0].delta.content
                                full_response_content += chunk_content
                                markdown_content += chunk_content
                                live.update(Markdown(markdown_content))
                    print()
                previous_responses.append(full_response_content)
                conversation.append({"role": "assistant", "content": full_response_content})
                match = re.search(r'```(\w+)?\n(.*?)```', full_response_content, re.DOTALL)
                if match:
                    language = match.group(1) or "python"
                    code = match.group(2)
                    console.print(Markdown(f"```{language}\n{code}```"))
                    filename = Prompt.ask("Enter the filename to save the code to (or press Enter to skip)")
                    if filename:
                        with open(filename, 'w', encoding='utf-8') as file:
                            file.write(code)
                        console.print(Panel(f"Code has been written to [success]{filename}[/success]", title="File Saved"))
            except Exception as e:
                console.print(Panel(f"Error: {str(e)}, try again.", title="Error", style="bold red"))

def ask_code_mode():
    global provider, ROUTING_MODEL, start_time
    console.print(Panel("**Ask Code Mode Activated**", style="bold red"))
    conversation = []

    def get_operating_system():
        """Detect the operating system for platform-specific commands."""
        if os.name == "nt":
            return "Windows"
        elif "TERMUX_VERSION" in os.environ:
            return "Termux"
        elif sys.platform == "darwin":
            return "MacOS"
        elif sys.platform.startswith("linux"):
            return "Linux"
        return "Unknown"

    def shell_agent_mode(conversation: list, file_path: str = "", code_content: str = ""):
        """Shell agent submode for generating and executing shell commands."""
        os_name = get_operating_system()
        shell_agent = Agent(client=client, system=f"""
        You are a shell command generator and executor for Ask Code Mode.
        The current working directory is: {os.getcwd()}.
        The operating system is: {os_name}.
        {'The current code file path is: ' + file_path if file_path else 'No code file loaded yet.'}
        {'The loaded code content is:' + code_content[:2000] if code_content else 'No code content loaded yet.'}
        Based on the user's query, generate shell commands relevant to the current directory or code context.
        Provide commands as plain text without backticks or language identifiers.
        - For Windows, use CMD commands (e.g., 'dir /b *.py *.js *.cpp') combining extensions into a single line when listing multiple file types.
        - For Linux, MacOS, or Termux, use Unix commands (e.g., 'find . -name "*.py" -o -name "*.js"').
        For multi-line scripts, provide the full script as a single block of text with proper separators (e.g., '&&' for Windows, ';' for Unix).
        Do not execute commands; only generate them for user approval.
        If the query is unclear, ask for clarification.
        """)
        console.print(Panel(f"Entered Shell Agent Mode (OS: {os_name}). Request shell commands or type 'quit-shell' to exit.", style="bold green"))

        def clean_shell_commands(commands: str) -> str:
            """Clean shell commands by removing Markdown markers."""
            commands = re.sub(r'```(?:sh|bash|cmd)?\s*', '', commands, flags=re.MULTILINE)
            commands = commands.replace('```', '').strip()
            return commands

        def execute_shell_script(script: str) -> bool:
            """Execute a shell script or command based on OS."""
            if os_name == "Windows":
                temp_script = os.path.join(os.getcwd(), "temp_script.bat")
                try:
                    # Combine dir commands into one line for Windows if multiple dir /b lines exist
                    lines = script.splitlines()
                    if all(line.strip().startswith("dir /b") for line in lines if line.strip()):
                        combined = " ".join(line.strip() for line in lines if line.strip())
                        script = combined
                    with open(temp_script, 'w', encoding='utf-8') as f:
                        f.write(f"@echo off\ncd /d \"{os.getcwd()}\"\n{script}")
                    cmd = f"cmd.exe /c \"{temp_script}\""
                    console.print(Panel(f"Executing shell script:\n{script}", style="cyan"))
                    consent = Prompt.ask("Execute this script? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                    if consent == "y":
                        try:
                            result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                            console.print(Panel(f"Command output:\n{result}", style="green"))
                            return True
                        except subprocess.CalledProcessError as e:
                            console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                            return False
                    return False
                finally:
                    if os.path.exists(temp_script):
                        os.remove(temp_script)
            else:  # Unix-like
                temp_script = os.path.join(os.getcwd(), "temp_script.sh")
                try:
                    with open(temp_script, 'w', encoding='utf-8') as f:
                        f.write(f"#!/bin/bash\ncd \"{os.getcwd()}\"\n{script}")
                    os.chmod(temp_script, 0o755)
                    cmd = f"bash \"{temp_script}\""
                    console.print(Panel(f"Executing shell script:\n{script}", style="cyan"))
                    consent = Prompt.ask("Execute this script? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                    if consent == "y":
                        try:
                            result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                            console.print(Panel(f"Command output:\n{result}", style="green"))
                            return True
                        except subprocess.CalledProcessError as e:
                            console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                            return False
                    return False
                finally:
                    if os.path.exists(temp_script):
                        os.remove(temp_script)

        while True:
            shell_query = input("Shell query (or 'quit-shell' to exit): ").strip()
            if shell_query.lower() == "quit-shell":
                console.print(Panel("Exited Shell Agent Mode", style="bold green"))
                break
            conversation.append({"role": "user", "content": shell_query})
            try:
                shell_commands = shell_agent(shell_query)
                cleaned_commands = clean_shell_commands(shell_commands)
                console.print(Panel(cleaned_commands, title="Generated Shell Commands", style="cyan"))
                conversation.append({"role": "assistant", "content": cleaned_commands})

                exec_choice = Prompt.ask("Execute commands? ([a]ll, [s]tep-by-step, [n]o)", choices=["a", "s", "n"], default="n", console=console)
                command_lines = cleaned_commands.splitlines()

                if exec_choice == "a":
                    if len(command_lines) > 1 or any(line.strip().startswith("for") or line.strip().startswith("#!") for line in command_lines):
                        if not execute_shell_script(cleaned_commands):
                            console.print(Panel("Script execution failed or skipped.", style="yellow"))
                    else:
                        for cmd in command_lines:
                            if cmd.strip():
                                console.print(Panel(f"Executing shell command:\n{cmd}", style="cyan"))
                                consent = Prompt.ask("Execute this command? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                                if consent == "y":
                                    try:
                                        result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                                        console.print(Panel(f"Command output:\n{result}", style="green"))
                                    except subprocess.CalledProcessError as e:
                                        console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                                        console.print(Panel("Command execution failed or skipped.", style="yellow"))
                                        break
                elif exec_choice == "s":
                    for cmd in command_lines:
                        if cmd.strip():
                            console.print(Panel(f"Next command: {cmd}", style="cyan"))
                            consent = Prompt.ask("Execute this command? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                            if consent == "y":
                                try:
                                    result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                                    console.print(Panel(f"Command output:\n{result}", style="green"))
                                except subprocess.CalledProcessError as e:
                                    console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                                    console.print(Panel("Command execution stopped.", style="yellow"))
                                    break
            except Exception as e:
                console.print(Panel(f"Error generating or executing shell commands: {e}", style="red"))

    while True:
        file_path_prompt = f"Enter the path to your code file (or 'quit' to exit, 'save' to save conversation, 'shell' for shell commands): "
        file_path = Prompt.ask(file_path_prompt)
        conversation.append({"role": "user", "content": file_path_prompt + file_path})

        if file_path.lower() == 'quit':
            console.print(Panel("**Exited Ask Code Mode**", style="bold red"))
            break
        elif file_path.lower() == "save":
            if conversation:
                filename = Prompt.ask("Enter the filename to save the conversation to (e.g., my_conversation.md) or press Enter to skip")
                if filename and save_conversation(conversation, filename, provider, ROUTING_MODEL, start_time):
                    if operating_system == "Termux":
                        handle_termux_file_options(filename)
            continue
        elif file_path.lower() == "shell":
            shell_agent_mode(conversation)
            continue

        file_path = file_path.strip('"').strip("'")
        if not os.path.isfile(file_path):
            console.print(Markdown(f"**Error: File '{file_path}' not found.**"))
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                code_content = file.read()
        except UnicodeDecodeError:
            console.print(Markdown(f"**Error: Could not decode file '{file_path}' with UTF-8.**"))
            continue
        except Exception as e:
            console.print(Markdown(f"**Error reading file '{file_path}': {e}**"))
            continue

        previous_responses = []
        query_prompt = "What would you like to know about this code?\n\n"
        console.print(query_prompt)

        while True:
            user_query_prompt = f"Enter your query (Type 'new' to query a new code file, 'quit' to exit, 'save' to save conversation, 'shell' for shell commands): "
            user_query = Prompt.ask(Text(user_query_prompt, style="green"))
            conversation.append({"role": "user", "content": user_query_prompt + user_query})

            if user_query.lower() == 'new':
                console.print("[bold]Starting new code query...[/bold]")
                break
            elif user_query.lower() == 'quit':
                console.print(Panel("**Exited Ask Code Mode**", style="bold red"))
                return
            elif user_query.lower() == "save":
                if conversation:
                    filename = Prompt.ask("Enter the filename to save the conversation to (e.g., my_conversation.md) or press Enter to skip")
                    if filename and save_conversation(conversation, filename, provider, ROUTING_MODEL, start_time):
                        if operating_system == "Termux":
                            handle_termux_file_options(filename)
                continue
            elif user_query.lower() == "shell":
                shell_agent_mode(conversation, file_path, code_content)
                continue

            full_query = "\n".join(previous_responses) + "\n" + user_query
            try:
                if provider == "Anthropic":
                    messages = [{"role": "user", "content": [{"type": "text", "text": full_query}, {"type": "text", "text": f"Code:\n```\n{code_content}\n```"}]}]
                    response = client.messages.create(model=GENERAL_MODEL, max_tokens=8000, messages=messages, stream=True)
                    full_response_content = ""
                    if response.content:
                        for part in response.content:
                            if hasattr(part, "text"):
                                full_response_content += part.text
                    markdown_content = ""
                    with Live(console=console, refresh_per_second=4) as live:
                        for char in full_response_content:
                            markdown_content += char
                            live.update(Markdown(markdown_content))
                            time.sleep(0.02)
                    print()
                elif provider == "Mistral":
                    messages = [{"role": "user", "content": [{"type": "text", "text": full_query}, {"type": "text", "text": f"Code:\n```\n{code_content}\n```"}]}]
                    response = client.chat.stream(model=GENERAL_MODEL, messages=messages)
                    full_response_content = ""
                    markdown_content = ""
                    with Live(console=console, refresh_per_second=4) as live:
                        for chunk in response:
                            if chunk.data.choices[0].delta.content is not None:
                                chunk_content = chunk.data.choices[0].delta.content
                                full_response_content += chunk_content
                                markdown_content += chunk_content
                                live.update(Markdown(markdown_content))
                    print()
                else:
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": [{"type": "text", "text": full_query}, {"type": "text", "text": f"Code:\n```\n{code_content}\n```"}]}],
                        model=GENERAL_MODEL,
                        stream=True
                    )
                    full_response_content = ""
                    markdown_content = ""
                    with Live(console=console, refresh_per_second=4) as live:
                        for chunk in response:
                            if chunk.choices and chunk.choices[0].delta.content:
                                chunk_content = chunk.choices[0].delta.content
                                full_response_content += chunk_content
                                markdown_content += chunk_content
                                live.update(Markdown(markdown_content))
                    print()
                previous_responses.append(full_response_content)
                conversation.append({"role": "assistant", "content": full_response_content})
                match = re.search(r'```(\w+)?\n(.*?)```', full_response_content, re.DOTALL)
                if match:
                    language = match.group(1) or "python"
                    code = match.group(2)
                    console.print(Markdown(f"```{language}\n{code}```"))
                    filename = Prompt.ask("Enter the filename to save the code to (or press Enter to skip)")
                    if filename:
                        with open(filename, 'w', encoding='utf-8') as file:
                            file.write(code)
                        console.print(Panel(f"Code has been written to [success]{filename}[/success]", title="File Saved"))
            except Exception as e:
                console.print(Panel(f"Error: {str(e)}, try again.", title="Error", style="bold red"))


from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdf2image import convert_from_path
from io import BytesIO

def ask_document_mode():
    global provider, ROUTING_MODEL, start_time
    console.print(Panel("**Ask Document Mode Activated**", style="bold red"))
    conversation = []

    def get_operating_system():
        """Detect the operating system for platform-specific commands."""
        if os.name == "nt":
            return "Windows"
        elif "TERMUX_VERSION" in os.environ:
            return "Termux"
        elif sys.platform == "darwin":
            return "MacOS"
        elif sys.platform.startswith("linux"):
            return "Linux"
        return "Unknown"

    def shell_agent_mode(conversation: list, document_text: str = "", image_base64_strings: list = []):
        """Shell agent submode for generating and executing shell commands."""
        os_name = get_operating_system()
        shell_agent = Agent(client=client, system=f"""
        You are a shell command generator and executor for Ask Document Mode.
        The current working directory is: {os.getcwd()}.
        The operating system is: {os_name}.
        {'The loaded document content is:' + document_text[:2000] if document_text else 'No document loaded yet.'}
        {'Images are present in the document.' if image_base64_strings else 'No images in document.'}
        Based on the user's query, generate shell commands relevant to the current directory or document context.
        Provide commands as plain text without backticks or language identifiers.
        - For Windows, use CMD commands (e.g., 'dir /b *.pdf') or PowerShell if specified.
        - For Linux, MacOS, or Termux, use Unix commands (e.g., 'find . -name "*.pdf"').
        For multi-line scripts, provide the full script as a single block of text.
        Do not execute commands; only generate them for user approval.
        If the query is unclear, ask for clarification.
        """)
        console.print(Panel(f"Entered Shell Agent Mode (OS: {os_name}). Request shell commands or type 'quit-shell' to exit.", style="bold green"))

        def clean_shell_commands(commands: str) -> str:
            """Clean shell commands by removing Markdown markers."""
            commands = re.sub(r'```(?:sh|bash|cmd)?\s*', '', commands, flags=re.MULTILINE)
            commands = commands.replace('```', '').strip()
            return commands

        def execute_shell_script(script: str) -> bool:
            """Execute a shell script or command based on OS."""
            if os_name == "Windows":
                temp_script = os.path.join(os.getcwd(), "temp_script.bat")
                try:
                    with open(temp_script, 'w', encoding='utf-8') as f:
                        f.write(script)
                    cmd = f"cmd.exe /c {temp_script}"
                    console.print(Panel(f"Executing shell script:\n{script}", style="cyan"))
                    consent = Prompt.ask("Execute this script? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                    if consent == "y":
                        try:
                            result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                            console.print(Panel(f"Command output:\n{result}", style="green"))
                            return True
                        except subprocess.CalledProcessError as e:
                            console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                            return False
                    return False
                finally:
                    if os.path.exists(temp_script):
                        os.remove(temp_script)
            else:  # Unix-like
                temp_script = os.path.join(os.getcwd(), "temp_script.sh")
                try:
                    with open(temp_script, 'w', encoding='utf-8') as f:
                        f.write("#!/bin/bash\n" + script)
                    os.chmod(temp_script, 0o755)
                    cmd = f"bash {temp_script}"
                    console.print(Panel(f"Executing shell script:\n{script}", style="cyan"))
                    consent = Prompt.ask("Execute this script? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                    if consent == "y":
                        try:
                            result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                            console.print(Panel(f"Command output:\n{result}", style="green"))
                            return True
                        except subprocess.CalledProcessError as e:
                            console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                            return False
                    return False
                finally:
                    if os.path.exists(temp_script):
                        os.remove(temp_script)

        while True:
            shell_query = input("Shell query (or 'quit-shell' to exit): ").strip()
            if shell_query.lower() == "quit-shell":
                console.print(Panel("Exited Shell Agent Mode", style="bold green"))
                break
            conversation.append({"role": "user", "content": shell_query})
            try:
                shell_commands = shell_agent(shell_query)
                cleaned_commands = clean_shell_commands(shell_commands)
                console.print(Panel(cleaned_commands, title="Generated Shell Commands", style="cyan"))
                conversation.append({"role": "assistant", "content": cleaned_commands})

                exec_choice = Prompt.ask("Execute commands? ([a]ll, [s]tep-by-step, [n]o)", choices=["a", "s", "n"], default="n", console=console)
                command_lines = cleaned_commands.splitlines()

                if exec_choice == "a":
                    if len(command_lines) > 1 or any(line.strip().startswith("for") or line.strip().startswith("#!") for line in command_lines):
                        if not execute_shell_script(cleaned_commands):
                            console.print(Panel("Script execution failed or skipped.", style="yellow"))
                    else:
                        for cmd in command_lines:
                            if cmd.strip():
                                console.print(Panel(f"Executing shell command:\n{cmd}", style="cyan"))
                                consent = Prompt.ask("Execute this command? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                                if consent == "y":
                                    try:
                                        result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                                        console.print(Panel(f"Command output:\n{result}", style="green"))
                                    except subprocess.CalledProcessError as e:
                                        console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                                        console.print(Panel("Command execution failed or skipped.", style="yellow"))
                                        break
                elif exec_choice == "s":
                    if len(command_lines) > 1 or any(line.strip().startswith("for") or line.strip().startswith("#!") for line in command_lines):
                        console.print(Panel(f"Script to execute step-by-step:\n{cleaned_commands}", style="cyan"))
                        if not execute_shell_script(cleaned_commands):
                            console.print(Panel("Script execution stopped.", style="yellow"))
                    else:
                        for cmd in command_lines:
                            if cmd.strip():
                                console.print(Panel(f"Next command: {cmd}", style="cyan"))
                                consent = Prompt.ask("Execute this command? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                                if consent == "y":
                                    try:
                                        result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                                        console.print(Panel(f"Command output:\n{result}", style="green"))
                                    except subprocess.CalledProcessError as e:
                                        console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                                        console.print(Panel("Command execution stopped.", style="yellow"))
                                        break
            except Exception as e:
                console.print(Panel(f"Error generating or executing shell commands: {e}", style="red"))

    def execute_command_with_consent(cmd: str, description: str):
        """Execute a command with user consent."""
        console.print(Panel(f"{description}:\n{cmd}", style="cyan"))
        consent = Prompt.ask("Execute this command? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
        if consent == "y":
            try:
                result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                console.print(Panel(f"Command output:\n{result}", style="green"))
                return True
            except subprocess.CalledProcessError as e:
                console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                return False
        return False

    while True:
        file_path_prompt = f"Enter the path to your document file (PDF, Markdown, or text) (or 'quit' to exit, 'save' to save conversation, 'shell' for shell commands): "
        file_path = Prompt.ask(file_path_prompt)
        conversation.append({"role": "user", "content": file_path_prompt + file_path})

        if file_path.lower() == 'quit':
            console.print(Markdown("**Exited Ask Document Mode**"))
            break
        elif file_path.lower() == "save":
            if conversation:
                filename = Prompt.ask("Enter the filename to save the conversation to (e.g., my_conversation.md) or press Enter to skip")
                if filename and save_conversation(conversation, filename, provider, ROUTING_MODEL, start_time):
                    if operating_system == "Termux":
                        handle_termux_file_options(filename)
            continue
        elif file_path.lower() == "shell":
            shell_agent_mode(conversation)
            continue

        file_path = file_path.strip('"').strip("'")
        if not os.path.isfile(file_path):
            console.print(Markdown(f"**Error: File '{file_path}' not found.**"))
            continue

        file_ext = os.path.splitext(file_path)[1].lower()
        document_text = ""
        image_base64_strings = []

        if file_ext == '.pdf':
            try:
                with open(file_path, 'rb') as fp:
                    parser = PDFParser(fp)
                    document = PDFDocument(parser)
                    rsrcmgr = PDFResourceManager()
                    laparams = LAParams()
                    output_string = StringIO()
                    device = TextConverter(rsrcmgr, output_string, laparams=laparams)
                    interpreter = PDFPageInterpreter(rsrcmgr, device)
                    for page in PDFPage.create_pages(document):
                        interpreter.process_page(page)
                    document_text = output_string.getvalue()
            except Exception as e:
                console.print(Markdown(f"**Error processing PDF '{file_path}': {e}**"))
                continue
            try:
                pdf_images = convert_from_path(file_path, fmt="jpeg")
                for img in pdf_images:
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    image_base64_strings.append(img_str)
            except Exception as e:
                console.print(Markdown("**(optional) To query images in document, install poppler. See 'install help'.**"))
        elif file_ext == '.md':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    document_text = f.read()
            except Exception as e:
                console.print(Markdown(f"**Error reading Markdown file '{file_path}': {e}**"))
                continue
        elif file_ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    document_text = f.read()
            except Exception as e:
                console.print(Markdown(f"**Error reading text file '{file_path}': {e}**"))
                continue
        else:
            console.print(Markdown(f"**Error: Unsupported file type '{file_ext}'.**"))
            continue

        if image_base64_strings:
            for idx, img_str in enumerate(image_base64_strings):
                conversation.append({"role": "user", "content": f"Image {idx+1} in the document: \n```\n<img src='data:image/jpeg;base64,{img_str}'/>\n```"})

        previous_responses = []
        query_prompt = "What would you like to know about this document?\n\n"
        console.print(query_prompt)

        while True:
            user_query_prompt = f"Enter your query (Type 'new' to query a new document, 'quit' to exit, 'save' to save conversation, 'shell' for shell commands): "
            user_query = Prompt.ask(Text(user_query_prompt, style="green"))
            conversation.append({"role": "user", "content": user_query_prompt + user_query})

            if user_query.lower() == 'new':
                console.print("[bold]Starting new document query...[/bold]")
                break
            elif user_query.lower() == 'quit':
                console.print(Panel("**Exited Ask Document Mode**", style="bold red"))
                return
            elif user_query.lower() == "save":
                if conversation:
                    filename = Prompt.ask("Enter the filename to save the conversation to (e.g., my_conversation.md) or press Enter to skip")
                    if filename and save_conversation(conversation, filename, provider, ROUTING_MODEL, start_time):
                        if operating_system == "Termux":
                            handle_termux_file_options(filename)
                continue
            elif user_query.lower() == "shell":
                shell_agent_mode(conversation, document_text, image_base64_strings)
                continue

            full_query = "\n".join(previous_responses) + "\n" + user_query
            try:
                messages = []
                if document_text:
                    messages.append({"role": "user", "content": [{"type": "text", "text": full_query}, {"type": "text", "text": f"Document Content:\n```\n{document_text}\n```"}]})
                if image_base64_strings:
                    for img_str in image_base64_strings:
                        messages.append({"role": "user", "content": [{"type": "text", "text": full_query}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}]})

                if provider == "Anthropic":
                    response = client.messages.create(model=GENERAL_MODEL, max_tokens=1024, messages=messages, stream=True)
                    full_response_content = ""
                    if response.content:
                        for part in response.content:
                            if hasattr(part, "text"):
                                full_response_content += part.text
                    markdown_content = ""
                    with Live(console=console, refresh_per_second=4) as live:
                        for char in full_response_content:
                            markdown_content += char
                            live.update(Markdown(markdown_content))
                            time.sleep(0.02)
                    print()
                elif provider == "Mistral":
                    response = client.chat.stream(model=GENERAL_MODEL, messages=messages)
                    full_response_content = ""
                    markdown_content = ""
                    with Live(console=console, refresh_per_second=4) as live:
                        for chunk in response:
                            if chunk.data.choices[0].delta.content is not None:
                                chunk_content = chunk.data.choices[0].delta.content
                                full_response_content += chunk_content
                                markdown_content += chunk_content
                                live.update(Markdown(markdown_content))
                    print()
                else:
                    response = client.chat.completions.create(messages=messages, model=GENERAL_MODEL, stream=True)
                    full_response_content = ""
                    markdown_content = ""
                    with Live(console=console, refresh_per_second=4) as live:
                        for chunk in response:
                            if chunk.choices and chunk.choices[0].delta.content:
                                chunk_content = chunk.choices[0].delta.content
                                full_response_content += chunk_content
                                markdown_content += chunk_content
                                live.update(Markdown(markdown_content))
                    print()

                previous_responses.append(full_response_content)
                conversation.append({"role": "assistant", "content": full_response_content})
                match = re.search(r'```(\w+)?\n(.*?)```', full_response_content, re.DOTALL)
                if match:
                    language = match.group(1) or "python"
                    code = match.group(2)
                    console.print(Markdown(f"```{language}\n{code}```"))
                    filename = Prompt.ask("Enter the filename to save the code to (or press Enter to skip)")
                    if filename:
                        with open(filename, 'w', encoding='utf-8') as file:
                            file.write(code)
                        console.print(Panel(f"Code has been written to [success]{filename}[/success]", title="File Saved"))
            except Exception as e:
                console.print(Panel(f"Error: {str(e)}, try again.", title="Error", style="bold red"))


def create_web_app_generator(client):
    return Agent(client=client, system="""
    You are a web app generator. When a user asks for a web app, generate the code directly without any reasoning or observation.
    Provide the HTML and JavaScript code. Use Tailwind CSS classes for styling directly in the HTML. Do not include separate CSS code block, but use tailwind classes directly in the html.  Do not include tailwind script tag.
    Provide the HTML and JavaScript code in separate markdown code blocks with the appropriate language syntax highlighting.
    """)
web_app_generator = create_web_app_generator(client)

def web_app():
    global web_app_generator, provider, ROUTING_MODEL, start_time
    conversation = []
    server_process = None  # To track the running server process

    def run_server(filename):
        """Run a Python HTTP server to serve the generated HTML file."""
        nonlocal server_process
        if not os.path.isfile(filename):
            console.print(Panel(f"Error: File '{filename}' not found.", style="bold red"))
            return None
        os.chdir(os.path.dirname(os.path.abspath(filename)))  # Change to file's directory
        port = 8000  # Default port
        cmd = f"python -m http.server {port}"
        console.print(Panel(f"Starting server at http://localhost:{port}/ - Access {filename} in your browser.", style="white"))
        server_process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        console.print(Panel("Server running. Type 'stop' and press Enter to stop the server...", style="white"))
        return server_process

    def stop_server(process):
        """Stop the running Python HTTP server."""
        if process is None:
            console.print(Panel("No server is running.", style="yellow"))
            return
        if os.name == "nt":
            try:
                subprocess.run(f"taskkill /F /T /PID {process.pid}", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                console.print(Panel("Server terminated using taskkill.", style="green"))
            except subprocess.CalledProcessError as e:
                console.print(Panel(f"Failed to terminate server: {e}", style="yellow"))
        else:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                console.print(Panel("Server forcefully terminated.", style="yellow"))
        nonlocal server_process
        server_process = None

    def preview_server(process):
        """Handle server output and allow stopping via 'stop' command."""
        if process is None:
            return
        stop_event = threading.Event()
        output_lines = []

        def read_output():
            while not stop_event.is_set():
                output = process.stdout.readline()
                if output and output.strip():
                    output_lines.append(output.strip())
                    console.print(output.strip())
                if process.poll() is not None:
                    break
                time.sleep(0.01)

        output_thread = threading.Thread(target=read_output)
        output_thread.start()

        while True:
            stop_input = input().strip().lower()
            if stop_input == "stop":
                stop_event.set()
                stop_server(process)
                output_thread.join()
                console.print(Panel("Server stopped", style="green"))
                if output_lines:
                    console.print(Panel("\n".join(output_lines), title="Server Output", style="yellow"))
                break
            elif process.poll() is not None:
                stop_event.set()
                output_thread.join()
                console.print(Panel("Server stopped unexpectedly", style="yellow"))
                if output_lines:
                    console.print(Panel("\n".join(output_lines), title="Server Output", style="yellow"))
                break

    while True:
        console.print(Panel("**Web App Generation Mode**", title="Web App Generation Mode", style="bold red"))
        query_prompt = f"Enter your web app query (or 'quit' to exit, 'save' to save conversation, 'stop' to stop server if running): "
        query = Prompt.ask(Text(query_prompt, style="green"))
        conversation.append({"role": "user", "content": query_prompt + query})

        if query.lower() == 'quit':
            if server_process:
                stop_server(server_process)
            console.print(Panel("**Exited Web App Generation Mode**", style="bold red"))
            break
        elif query.lower() == "save":
            if conversation:
                filename = Prompt.ask("Enter the filename to save the conversation to (e.g., my_conversation.md) or press Enter to skip")
                if filename and save_conversation(conversation, filename, provider, ROUTING_MODEL, start_time):
                    if operating_system == "Termux":
                        handle_termux_file_options(filename)
            continue
        elif query.lower() == "stop":
            if server_process:
                stop_server(server_process)
            else:
                console.print(Panel("No server is running to stop.", style="yellow"))
            continue

        try:
            full_response_content = web_app_generator(query)
            conversation.append({"role": "assistant", "content": full_response_content})
            html_match = re.search(r'```html\n(.*?)```', full_response_content, re.DOTALL)
            js_match = re.search(r'```javascript\n(.*?)```', full_response_content, re.DOTALL)
            html_code = html_match.group(1) if html_match else ""
            js_code = js_match.group(1) if js_match else ""

            if html_code:
                console.print(Markdown(f"```html\n{html_code}```"))
            if js_code:
                console.print(Markdown(f"```javascript\n{js_code}```"))
            if not html_code and not js_code:
                console.print(Markdown(full_response_content))

            filename = Prompt.ask("Enter the filename to save the web app to (or press Enter to skip) (e.g., index.html)")
            if filename:
                try:
                    full_html = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Web App</title>
                    <script src="https://cdn.tailwindcss.com"></script>
                    </head>
                    <body>
                    {html_code}
                    <script>
                     {js_code}
                    </script>
                    </body>
                    </html>
                    """
                    with open(filename, 'w', encoding='utf-8') as file:
                        file.write(full_html)
                    console.print(Panel(f"Web app has been written to [success]{filename}[/success]", title="File Saved"))

                    if operating_system == "Termux":
                        handle_termux_file_options(filename)

                    # Option to run the server
                    run_choice = Prompt.ask("Would you like to run this web app using a Python server? ([y]es/[n]o)", choices=["y", "n"], default="n")
                    if run_choice == "y":
                        if server_process:
                            stop_server(server_process)  # Stop any existing server
                        server_process = run_server(filename)
                        if server_process:
                            preview_server(server_process)
                except Exception as e:
                    console.print(Markdown(f"**Error saving file: {e}**"))
        except Exception as e:
            console.print(Panel(f"Error: {str(e)}, try again.", title="Error", style="bold red"))

from PIL import Image
from huggingface_hub import InferenceClient

def image(client, IMAGE_MODEL, image_api_key):
    console.print(Panel("[bold]Image Generation Mode Activated[/bold]", title="Image Generation Mode", style="bold red"))
    while True:
        text_prompt = Prompt.ask(Text(f"Enter the text prompt for the image (or 'quit' to exit)", style="green"))
        if text_prompt.lower() == 'quit':
            console.print(Panel("**Exited Image Generation Mode**", style="bold red"))
            break
        while True:
            OUTPUT_FILE_NAME = Prompt.ask("Enter the output file name for the image (e.g., image.png)", default="image.png")
            if OUTPUT_FILE_NAME.lower().endswith((".png", ".jpg", ".jpeg")):
                break
            console.print("[warning]Please enter a valid file name with .png, .jpg, or .jpeg extension.[/warning]")
        MODEL_ID = IMAGE_MODEL
        HF_TOKEN = image_api_key
        try:
            image_client = InferenceClient(MODEL_ID, token=HF_TOKEN)
        except Exception as e:
            console.print(Panel(f"Error initializing image client: {e}", style="bold red"))
            continue
        start_time = time.time()
        progress_text = Text("Generating Image...", style="green")
        with Live(progress_text, console=console, refresh_per_second=10) as live:
            animation_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            index = 0
            try:
                while True:
                    try:
                        image = image_client.text_to_image(text_prompt)
                        break
                    except Exception:
                        index = (index + 1) % len(animation_chars)
                        progress_text.plain = f"Generating Image... {animation_chars[index]}"
                        live.update(progress_text)
                        time.sleep(0.1)
            except Exception as e:
                console.print(Panel(f"Error generating image: {e}", style="bold red"))
                continue
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(output_dir, OUTPUT_FILE_NAME)
        try:
            image.save(output_path)
            end_time = time.time()
            elapsed_time = end_time - start_time
            console.print(Panel(f"Image saved to: [success]{output_path}[/success]", title="Image Saved"))
            console.print(Panel(f"Image generation time: [bold]{elapsed_time:.2f} seconds[/bold]", title="Generation Time"))
            if operating_system == "Termux":
                handle_termux_file_options(output_path)
        except Exception as e:
            console.print(Panel(f"Error saving image: {e}", style="bold red"))
        if Prompt.ask("Do you want to generate another image? (yes/no)", default="no").lower() != "yes":
            console.print(Panel("**Exited Image Generation Mode**", style="bold red"))
            break





import shutil
from pathlib import Path
import time
import os
import re
import subprocess
import json
from typing import Optional, List, Dict
from rich.panel import Panel
from rich.prompt import Prompt
from rich.console import Console





def vitex_app_mode():
    console.print(Panel("**Vitex App Mode**", title="Vitex App Mode", style="bold red"))
    project_dir = None
    current_project_state = ""
    conversation = []
    backups = {}
    last_errors = []
    is_new_project = False
    first_query_processed = False

    def search_full_path(short_path: str) -> Optional[str]:
        """Search for the full path of a directory based on a short path."""
        search_dirs = [os.getcwd(), os.path.expanduser("~")]
        for base_dir in search_dirs:
            for root, dirs, _ in os.walk(base_dir):
                for dir_name in dirs:
                    if dir_name == short_path:
                        full_path = os.path.abspath(os.path.join(root, dir_name))
                        if os.path.isdir(full_path):
                            return full_path
        return None

    def suggest_node_npm_installation():
        """Suggest commands to check and install Node.js and npm."""
        os_name = get_operating_system()
        check_cmd = "node -v && npm -v"
        if os_name == "Windows":
            install_cmd = "Please download and install Node.js from https://nodejs.org/ manually."
        elif os_name in ("Linux", "Termux"):
            install_cmd = "apt update && apt install -y nodejs npm" if os_name == "Linux" else "pkg install nodejs"
        else:  # MacOS or other Unix
            install_cmd = "brew install node" if shutil.which("brew") else "curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && apt install -y nodejs"
        
        console.print(Panel(
            f"To check if Node.js and npm are installed, run:\n{check_cmd}\n\n"
            f"If not installed, install them with:\n{install_cmd}",
            title="Node.js and npm Setup",
            style="white"
        ))
        return check_cmd, install_cmd

    def suggest_framework_installation(framework: str):
        """Suggest commands to check and install framework-specific tools (only for Expo)."""
        if framework == "Expo":
            check_cmd = "npx create-expo-app --version"
            install_cmd = "npm install -g create-expo-app"
            console.print(Panel(
                f"To check if {framework} tool is installed, run:\n{check_cmd}\n\n"
                f"If not installed, install it with:\n{install_cmd}",
                title=f"{framework} Setup",
                style="white"
            ))
            return check_cmd, install_cmd
        return None, None  # No prompts for Vite

    def execute_command_with_consent(cmd: str, description: str):
        """Execute a command with user consent."""
        console.print(Panel(f"{description}:\n{cmd}", style="white"))
        consent = Prompt.ask("Execute this command? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
        if consent == "y":
            try:
                result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                console.print(Panel(f"Command output:\n{result}", style="green"))
                return True
            except subprocess.CalledProcessError as e:
                console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                return False
        return False

    def stop_process(process):
        """Stop the process using taskkill on Windows or SIGTERM on other OS."""
        if os.name == "nt":
            try:
                subprocess.run(f"taskkill /F /T /PID {process.pid}", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                console.print(Panel("Process terminated using taskkill.", style="green"))
            except subprocess.CalledProcessError as e:
                console.print(Panel(f"Failed to terminate process: {e}", style="yellow"))
        else:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                console.print(Panel("Process forcefully terminated.", style="yellow"))

    # Suggest Node.js and npm setup
    node_check_cmd, node_install_cmd = suggest_node_npm_installation()
    console.print(Panel("Please ensure Node.js and npm are installed before proceeding.", style="yellow"))
    check_node = Prompt.ask("Would you like to check Node.js and npm versions manually? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
    if check_node == "y":
        execute_command_with_consent(node_check_cmd, "Checking Node.js and npm versions")
    else:
        install_node = Prompt.ask("Would you like to install Node.js and npm manually? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
        if install_node == "y" and node_install_cmd.startswith("Please"):
            console.print(Panel(node_install_cmd, style="yellow"))
        elif install_node == "y":
            if not execute_command_with_consent(node_install_cmd, "Installing Node.js and npm"):
                console.print(Panel("Node.js and npm installation skipped or failed. You may proceed, but functionality might be limited.", style="yellow"))

    # Choice between new or existing project
    project_type = Prompt.ask("Do you want to [1] create a new project or [2] use an existing project? (1/2)", choices=["1", "2"], console=console)
    
    if project_type == "1":
        is_new_project = True
        project_name = Prompt.ask("Enter the project name", default="my-app", console=console)
        framework_prompt = Prompt.ask("Choose a framework: [1] Vite (React), [2] Expo (1/2)", choices=["1", "2"], console=console)
        
        if framework_prompt == "1":
            framework = "Vite (React)"
            main_file = "src/App.jsx"
            style_file = "src/App.css"
            # No Vite installation prompts
            commands = [f"npm create vite@latest {project_name} -- --template react", f"cd {project_name}"]
        elif framework_prompt == "2":
            framework = "Expo"
            expo_check_cmd, expo_install_cmd = suggest_framework_installation(framework)
            check_expo = Prompt.ask(f"Would you like to check if {framework} tool (create-expo-app) is installed? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
            if check_expo == "y":
                execute_command_with_consent(expo_check_cmd, f"Checking {framework} tool (create-expo-app)")
            else:
                install_expo = Prompt.ask(f"Would you like to install {framework} tool (create-expo-app) manually? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                if install_expo == "y":
                    if not execute_command_with_consent(expo_install_cmd, f"Installing {framework} tool (create-expo-app)"):
                        console.print(Panel(f"{framework} tool installation skipped or failed. Proceeding might fail.", style="yellow"))

            # Additional Expo options
            language = Prompt.ask(
                "Choose a language: [1] JavaScript (default), [2] TypeScript (1/2)",
                choices=["1", "2"], default="1", console=console
            )
            template = Prompt.ask(
                "Choose a template: [1] Blank (minimal, default), [2] Tabs (navigation-ready), [3] Bare (full control) (1/2/3)",
                choices=["1", "2", "3"], default="1", console=console
            )

            # Set main_file based on language
            main_file = "App.js" if language == "1" else "App.tsx"
            style_file = main_file

            # Map template and language to the correct Expo template name
            if language == "1":  # JavaScript
                template_name = "blank" if template == "1" else "tabs" if template == "2" else "bare-minimum"
            else:  # TypeScript
                if template == "1":
                    template_name = "blank-typescript"
                elif template == "2":
                    template_name = "tabs-typescript"
                else:  # Bare doesn't have a direct TS template
                    template_name = "bare-minimum"
                    ts_setup_commands = [
                        f"cd {project_name}",
                        "npm install --save-dev typescript @types/react @types/react-native",
                        "touch tsconfig.json"
                    ]

            console.print(Panel(
                f"Expo project setup:\n- Language: {'JavaScript' if language == '1' else 'TypeScript'}\n- Template: {template_name.capitalize()}\n- Main file: {main_file}",
                title="Expo Configuration",
                style="white"
            ))

            # Base command for Expo project creation
            commands = [f"npx create-expo-app {project_name} --template {template_name}"]
            if language == "2" and template == "3":  # Add TS setup for Bare
                commands.extend(ts_setup_commands)
            else:
                commands.append(f"cd {project_name}")

        console.print(Panel("\n".join(f"{i+1}. {cmd}" for i, cmd in enumerate(commands)), title="Generated Commands", style="green"))
        exec_choice = Prompt.ask("Execute commands? ([a]ll, [s]tep-by-step, [n]o)", choices=["a", "s", "n"], default="n", console=console)
        if exec_choice == "a":
            for cmd in commands:
                if not execute_command_with_consent(cmd, f"Executing project creation step"):
                    console.print(Panel("Project creation failed or skipped.", style="yellow"))
                    return
            project_dir = os.path.abspath(os.path.join(os.getcwd(), project_name))
        elif exec_choice == "s":
            for cmd in commands:
                console.print(Panel(f"Next command: {cmd}", style="white"))
                if not execute_command_with_consent(cmd, "Executing project creation step"):
                    console.print(Panel("Project creation stopped.", style="yellow"))
                    return
            project_dir = os.path.abspath(os.path.join(os.getcwd(), project_name))
        else:
            console.print(Panel("Setup cancelled", style="yellow"))
            return

        install_choice = Prompt.ask("Would you like to install initial dependencies now? ([y]es/[n]o)", choices=["y", "n"], default="y", console=console)
        if install_choice == "y":
            os.chdir(project_dir)
            if not execute_command_with_consent("npm install", "Installing initial dependencies"):
                console.print(Panel("Dependency installation failed or skipped.", style="yellow"))
                return

        if framework == "Expo":
            expo_web_deps = ["react-dom", "react-native-web", "@expo/metro-runtime"]
            console.print(Panel(f"Expo web support (enabled by default) requires additional dependencies: {', '.join(expo_web_deps)}", style="white"))
            for dep in expo_web_deps:
                install_prompt = Prompt.ask(f"Would you like to install {dep}? ([y]es/[n]o)", choices=["y", "n"], default="y", console=console)
                if install_prompt == "y":
                    os.chdir(project_dir)
                    if not execute_command_with_consent(f"npx expo install {dep}", f"Installing {dep}"):
                        console.print(Panel(f"Failed to install {dep}. Expo web preview may not work.", style="yellow"))
                else:
                    console.print(Panel(f"Skipped installing {dep}. Note: Expo web preview may not work without it.", style="yellow"))

        # Verify initial main file content
        main_file_path = os.path.join(project_dir, main_file)
        if os.path.exists(main_file_path):
            with open(main_file_path, 'r', encoding='utf-8') as f:
                initial_content = f.read()
                console.print(Panel(f"Initial content of {main_file}:\n{initial_content}", title="Main File Verification", style="green"))
                if initial_content.startswith("TypeScript") or not initial_content.strip().startswith("import"):
                    console.print(Panel(f"Warning: {main_file} contains unexpected content at the top. Cleaning it now.", style="yellow"))
                    cleaned_content = re.sub(r'^(TypeScript|JavaScript)\s*', '', initial_content, flags=re.MULTILINE).strip()
                    with open(main_file_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    console.print(Panel(f"Cleaned content of {main_file}:\n{cleaned_content}", title="Cleaned Main File", style="green"))

    elif project_type == "2":
        project_path = Prompt.ask("Enter the full path or short name of your existing project directory", console=console)
        if os.path.isabs(project_path):
            project_dir = os.path.abspath(project_path)
        else:
            full_path = search_full_path(project_path)
            if full_path:
                console.print(Panel(f"Found project at: {full_path}", title="Path Preview", style="white"))
                use_path = Prompt.ask(f"Use this path ({full_path})? ([y]es/[n]o)", choices=["y", "n"], default="y", console=console)
                if use_path == "y":
                    project_dir = full_path
                else:
                    console.print(Panel("Path rejected. Please provide a valid full path.", style="yellow"))
                    return
            else:
                console.print(Panel(f"Could not find '{project_path}' in common directories. Please provide a full path.", style="yellow"))
                return
        
        if not os.path.isdir(project_dir):
            console.print(Panel(f"Error: Directory {project_dir} does not exist.", style="red"))
            return
        
        if os.path.exists(os.path.join(project_dir, "vite.config.js")):
            framework = "Vite (React)"
            main_file = "src/App.jsx"
            style_file = "src/App.css"
        elif os.path.exists(os.path.join(project_dir, "app.json")) and "expo" in open(os.path.join(project_dir, "package.json")).read():
            framework = "Expo"
            if os.path.exists(os.path.join(project_dir, "App.tsx")):
                main_file = "App.tsx"
            else:
                main_file = "App.js"
            style_file = main_file
        else:
            framework_prompt = Prompt.ask("Could not auto-detect framework. Choose: [1] Vite (React), [2] Expo (1/2)", choices=["1", "2"], console=console)
            framework = "Vite (React)" if framework_prompt == "1" else "Expo"
            if framework == "Vite (React)":
                main_file = "src/App.jsx"
                style_file = "src/App.css"
            else:
                language = Prompt.ask(
                    "Is this Expo project using [1] JavaScript or [2] TypeScript? (1/2)",
                    choices=["1", "2"], default="1", console=console
                )
                main_file = "App.js" if language == "1" else "App.tsx"
                style_file = main_file

        console.print(Panel(f"Using existing project at {project_dir} with framework {framework} and main file {main_file}", style="green"))

    def ingest_project(dir_path: str, specific_files: Optional[list] = None) -> str:
        ignore_patterns = {'node_modules', '*.log', '.git', '*.min.js', '*.min.css'}
        tree = []
        contents = []
        total_size = 0
        for root, dirs, files in os.walk(dir_path):
            rel_root = os.path.relpath(root, dir_path)
            if any(fnmatch(rel_root, pattern) for pattern in ignore_patterns):
                continue
            for file in files:
                if any(file.endswith(ext) for ext in ['.js', '.jsx', '.css', '.json', '.ts', '.tsx']) and not any(fnmatch(file, pattern) for pattern in ignore_patterns):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, dir_path)
                    if specific_files and rel_path not in specific_files:
                        continue
                    size = os.path.getsize(file_path)
                    if size > 10 * 1024 * 1024:
                        continue
                    total_size += size
                    if total_size > 500 * 1024 * 1024:
                        break
                    tree.append(rel_path)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        contents.append(f"File: {rel_path}\n{'='*48}\n{content}\n")
                    except Exception as e:
                        contents.append(f"File: {rel_path}\n{'='*48}\nError reading file: {e}\n")
        tree_str = "Directory structure:\n" + "\n".join([f"├── {path}" for path in tree])
        result = f"{tree_str}\n\n{'='*48}\n" + "".join(contents)
        return result

    def backup_files(files_dict: dict, project_dir: str):
        for path in files_dict.keys():
            full_path = os.path.join(project_dir, path)
            if os.path.exists(full_path):
                backup_path = os.path.join(project_dir, f".backup_{path.replace('/', '_')}_{int(time.time())}")
                shutil.copy2(full_path, backup_path)
                backups[path] = backup_path
                console.print(Panel(f"Backed up {path} to {backup_path}", style="white"))

    def revert_files(project_dir: str):
        if not backups:
            console.print(Panel("No backups available to revert", style="yellow"))
            return
        for path, backup_path in list(backups.items()):
            full_path = os.path.join(project_dir, path)
            if os.path.exists(backup_path):
                try:
                    shutil.move(backup_path, full_path)
                    console.print(Panel(f"Reverted {path} from {backup_path}", style="green"))
                    del backups[path]
                except Exception as e:
                    console.print(Panel(f"Failed to revert {path}: {e}", style="red"))
            else:
                console.print(Panel(f"Backup {backup_path} not found for {path}", style="yellow"))
                del backups[path]

    def compute_filtered_state(current_project_state: str, project_dir: str, query: str = "error analysis") -> str:
        all_files = [line.split("File: ")[1].strip() for line in current_project_state.splitlines() if line.startswith("File: ")]
        relevant_files = set()
        keywords = [kw.lower() for kw in query.split()] + ["app", "background", "color", "icon", "position", "page", "title", "size", "calculator", "quiz", "styles", "ui"]
        
        app_js_path = os.path.join(project_dir, main_file)
        if any(k in query.lower() for k in ["app", "split", "pdf", "mobile", "component", "background"]):
            if os.path.exists(app_js_path):
                relevant_files.add(main_file)

        filtered_state = ""
        for path in relevant_files:
            if path in all_files:
                sections = current_project_state.split("\n" + "="*48 + "\n")
                for section in sections:
                    if section.startswith(f"File: {path}"):
                        filtered_state += section + "\n"
                        break
            else:
                full_path = os.path.join(project_dir, path)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f"File: {path}\n{'='*48}\n{f.read()}\n"
                        filtered_state += content
                except Exception as e:
                    console.print(Panel(f"Error reading {full_path}: {e}", style="yellow"))

        if not filtered_state:
            console.print(Panel(f"Warning: No relevant files detected in query. Using {main_file} as fallback.", style="yellow"))
            if os.path.exists(app_js_path):
                with open(app_js_path, 'r', encoding='utf-8') as f:
                    filtered_state = f"File: {main_file}\n{'='*48}\n{f.read()}\n"
            else:
                filtered_state = current_project_state[:2000]

        return filtered_state

    def clean_llm_output(content: str, is_diff: bool = False) -> str:
        """Enhanced cleaning to remove unwanted labels like 'TypeScript' at the top."""
        if isinstance(content, list):
            content = "\n".join(content)
        # Remove code block markers and labels
        content = re.sub(r'```(?:javascript|jsx|typescript|shell|bash)?\s*', '', content, flags=re.MULTILINE)
        content = content.replace("```", "")
        # Remove standalone "TypeScript" or "JavaScript" at the start
        content = re.sub(r'^(TypeScript|JavaScript)\s*', '', content, flags=re.MULTILINE)
        lines = content.splitlines()
        if is_diff:
            cleaned_lines = [line for line in lines if line.strip().startswith(("+", "-", "File: ", "Command: "))]
        else:
            cleaned_lines = [line.lstrip("+-") for line in lines if line.strip() and not line.strip().startswith(("File:", "Command:", "="*48))]
        return "\n".join(cleaned_lines).strip()

    def split_install_commands(command: str) -> list[str]:
        """Split large install commands into individual commands."""
        if command.startswith("npm install") or command.startswith("npx expo install"):
            parts = command.split()
            packages = parts[2:] if parts[0] == "npm" else parts[3:]
            base_cmd = "npm install" if parts[0] == "npm" else "npx expo install"
            return [f"{base_cmd} {pkg}" for pkg in packages]
        return [command]

    def get_multiline_query():
        console.print(Panel("Enter your query (type 'submit-prompt' on a new line to provide your query). Use 'run' to start server, 'stop' to stop server, 'revert' to undo last change, 'chat' to enter chat mode, 'shell' to enter Vitex Shell Agent, 'quit' to exit:", style="green"))
        query_lines = []
        while True:
            line = input()
            if line.lower() == "submit-prompt":
                break
            query_lines.append(line)
        return "\n".join(query_lines).strip()

    def chat_mode(project_dir: str, current_project_state: str, framework: str, main_file: str, conversation: list):
        current_project_state = ingest_project(project_dir)
        error_context = f"\nLast Preview Errors:\n{chr(10).join(last_errors)}" if last_errors else ""
        app_js_content = ""
        app_js_path = os.path.join(project_dir, main_file)
        if os.path.exists(app_js_path):
            with open(app_js_path, 'r', encoding='utf-8') as f:
                app_js_content = f.read()
        else:
            sections = current_project_state.split("\n" + "="*48 + "\n")
            for section in sections:
                if section.startswith(f"File: {main_file}"):
                    app_js_content = section.split("\n", 2)[2]
                    break
            if not app_js_content:
                app_js_content = "Unable to read main file content from project state."

        chat_agent = Agent(client=client, system=f"""
        You are a helpful assistant for chatting about a {framework} app located at {project_dir}.
        The current project state is:\n{current_project_state[:2000]}{error_context}\n
        The main file '{main_file}' content is:\n{app_js_content}\n
        The style file is '{style_file}' (for styling).
        Answer user questions about the app or assist with planning.
        Provide concise, accurate responses based on the project state and framework.
        """)
        console.print(Panel("Entered Chat Mode. Ask about the app or type 'quit-chat' to exit.", style="bold green"))
        
        while True:
            chat_query = input("Chat query (or 'quit-chat' to exit): ").strip()
            if chat_query.lower() == "quit-chat":
                console.print(Panel("Exited Chat Mode", style="bold green"))
                break
            conversation.append({"role": "user", "content": chat_query})
            response = chat_agent(chat_query)
            console.print(Panel(response, title="Chat Response", style="white"))
            conversation.append({"role": "assistant", "content": response})

    def vitex_shell_agent_mode(project_dir: str, current_project_state: str, framework: str, main_file: str, conversation: list):
        current_project_state = ingest_project(project_dir)
        error_context = f"\nLast Preview Errors:\n{chr(10).join(last_errors)}" if last_errors else ""
        app_js_content = ""
        app_js_path = os.path.join(project_dir, main_file)
        if os.path.exists(app_js_path):
            with open(app_js_path, 'r', encoding='utf-8') as f:
                app_js_content = f.read()
        else:
            sections = current_project_state.split("\n" + "="*48 + "\n")
            for section in sections:
                if section.startswith(f"File: {main_file}"):
                    app_js_content = section.split("\n", 2)[2]
                    break
            if not app_js_content:
                app_js_content = "Unable to read main file content from project state."

        shell_agent = Agent(client=client, system=f"""
        You are a shell command generator for a {framework} app located at {project_dir}.
        The current project state is:\n{current_project_state[:2000]}{error_context}\n
        The main file '{main_file}' content is:\n{app_js_content}\n
        The style file is '{style_file}' (for styling).
        Generate shell commands tailored to this {framework} project based on the user's query.
        Provide commands as plain text without backticks or language identifiers.
        """)
        console.print(Panel("Entered Vitex Shell Agent Mode. Request shell commands or type 'quit-shell' to exit.", style="bold green"))

        def clean_shell_commands(commands: str) -> str:
            commands = re.sub(r'```(?:sh|bash)?\s*', '', commands, flags=re.MULTILINE)
            commands = commands.replace('```', '').strip()
            return commands

        def execute_shell_script(script: str, project_dir: str) -> bool:
            os.chdir(project_dir)
            temp_script = os.path.join(project_dir, "temp_script.sh")
            try:
                with open(temp_script, 'w', encoding='utf-8') as f:
                    f.write("#!/bin/bash\n" + script)
                os.chmod(temp_script, 0o755)
                cmd = f"bash {temp_script}"
                console.print(Panel(f"Executing shell script:\n{script}", style="white"))
                consent = Prompt.ask("Execute this script? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                if consent == "y":
                    try:
                        result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                        console.print(Panel(f"Command output:\n{result}", style="green"))
                        return True
                    except subprocess.CalledProcessError as e:
                        console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                        return False
                return False
            finally:
                if os.path.exists(temp_script):
                    os.remove(temp_script)

        while True:
            shell_query = input("Shell query (or 'quit-shell' to exit): ").strip()
            if shell_query.lower() == "quit-shell":
                console.print(Panel("Exited Vitex Shell Agent Mode", style="bold green"))
                break
            conversation.append({"role": "user", "content": shell_query})
            try:
                shell_commands = shell_agent(shell_query)
                cleaned_commands = clean_shell_commands(shell_commands)
                console.print(Panel(cleaned_commands, title="Generated Shell Commands", style="white"))
                conversation.append({"role": "assistant", "content": cleaned_commands})

                exec_choice = Prompt.ask("Execute commands? ([a]ll, [s]tep-by-step, [n]o)", choices=["a", "s", "n"], default="n", console=console)
                os.chdir(project_dir)
                command_lines = cleaned_commands.splitlines()

                if exec_choice == "a":
                    if len(command_lines) > 1 or any(line.strip().startswith("for") or line.strip().startswith("#!") for line in command_lines):
                        if not execute_shell_script(cleaned_commands, project_dir):
                            console.print(Panel("Script execution failed or skipped.", style="yellow"))
                    else:
                        for cmd in command_lines:
                            if cmd.strip() and not execute_command_with_consent(cmd.strip(), "Executing shell command"):
                                console.print(Panel("Command execution failed or skipped.", style="yellow"))
                                break
                elif exec_choice == "s":
                    if len(command_lines) > 1 or any(line.strip().startswith("for") or line.strip().startswith("#!") for line in command_lines):
                        console.print(Panel(f"Script to execute step-by-step:\n{cleaned_commands}", style="white"))
                        if not execute_shell_script(cleaned_commands, project_dir):
                            console.print(Panel("Script execution stopped.", style="yellow"))
                    else:
                        for cmd in command_lines:
                            if cmd.strip():
                                console.print(Panel(f"Next command: {cmd}", style="white"))
                                if not execute_command_with_consent(cmd.strip(), "Executing shell command"):
                                    console.print(Panel("Command execution stopped.", style="yellow"))
                                    break
            except Exception as e:
                console.print(Panel(f"Error generating or executing shell commands: {e}", style="red"))

    def run_app(project_dir: str, framework: str) -> subprocess.Popen:
        os.chdir(project_dir)
        cmd = "npm run dev" if framework == "Vite (React)" else "npx expo start --web"
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process

    def preview_app(process: subprocess.Popen):
        console.print(Panel("App running. Type 'stop' and press Enter to stop the app and continue editing...", style="white"))
        output_lines = []
        stop_event = threading.Event()

        def read_output():
            while not stop_event.is_set():
                output = process.stdout.readline()
                if output and output.strip():
                    output_lines.append(output.strip())
                    console.print(output.strip())
                if process.poll() is not None:
                    break
                time.sleep(0.01)

        output_thread = threading.Thread(target=read_output)
        output_thread.start()

        while True:
            stop_input = input().strip().lower()
            if stop_input == "stop":
                stop_event.set()
                stop_process(process)
                output_thread.join()
                console.print(Panel("App stopped", style="green"))
                if output_lines:
                    console.print(Panel("\n".join(output_lines), title="Terminal Output - Copy these for modification", style="yellow"))
                    return output_lines
                break
            elif process.poll() is not None:
                stop_event.set()
                output_thread.join()
                console.print(Panel("App stopped unexpectedly", style="yellow"))
                if output_lines:
                    console.print(Panel("\n".join(output_lines), title="Terminal Output", style="yellow"))
                    return output_lines
                break

    def refine_query(original_query: str, llm_agent, is_first_query: bool = False) -> str:
        if is_new_project and is_first_query:
            refine_agent = Agent(client=client, system="You are a query refiner for a new project setup. Analyze the user's query and generate a detailed, single-paragraph refined query that includes comprehensive features based on the original intent. Return only the refined query without explanation.")
            return refine_agent(f"Refine this query: '{original_query}'")
        else:
            refine_agent = Agent(client=client, system="You are a query refiner. Analyze the user's query and provide a detailed, equivalent refined version that enhances clarity and specificity while maintaining the original intent. Return only the refined query without explanation.")
            return refine_agent(f"Refine this query: '{original_query}'")

    def process_llm_response(query: str, filtered_state: str, framework: str, project_dir: str, conversation: list, llm_agent) -> tuple[Optional[str], Optional[list[str]]]:
        error_context = f"\nLast Preview Errors:\n{chr(10).join(last_errors)}" if last_errors else ""
        key_project_state = ""
        sections = current_project_state.split("\n" + "="*48 + "\n")
        for section in sections:
            if section.startswith(f"File: {main_file}"):
                key_project_state += section + "\n"
                break
        if not key_project_state:
            key_project_state = f"File: {main_file} not found in project state."

        llm_agent.messages[0]["content"] = (
            f"Here is the current state of key project files (e.g., {main_file}):\n{key_project_state}{error_context}\n" +
            f"Full project state (use this if filtered state is insufficient):\n{current_project_state[:2000]}\n" +
            f"Filtered state for query (use this as the base code to modify):\n{filtered_state}\n" +
            f"Full project directory: {project_dir}\n" +
            f"Framework: {framework}\n" +
            f"Assist with modifications based on user request: '{query}'. First, provide a modification plan as plain text under 'Modification Plan:' explaining what files will be changed, what shell commands (e.g., npm install) are needed, and how the app will be modified. " +
            "After approval, generate the COMPLETE updated file content under 'File: <path>' headers and shell commands under 'Command: <cmd>' headers, without diff markers ('+' or '-'), language identifiers (e.g., 'javascript', 'shell'), or backticks (```). " +
            "For each shell command (e.g., installing a package), provide a separate 'Command: <cmd>' line. Do not combine multiple installations into a single command; list them individually (e.g., 'Command: npx expo install expo-document-picker', 'Command: npx expo install pdf-lib'). " +
            "For new app features (e.g., 'create a PDF splitter mobile app'), generate necessary code and commands (e.g., 'Command: npx expo install pdf-lib'). " +
            "Preserve existing functionality unless explicitly requested to change. " +
            f"For UI/feature changes, target '{main_file}'. For styling, modify StyleSheet in '{main_file}' (Expo) or '{style_file}' (Vite). " +
            "Do not include comments or explanatory text in the code/command output—only the full content as specified."
        )
        try:
            modification_plan_response = llm_agent(query)
            console.print(Panel(modification_plan_response, title="AI Modification Plan", style="white"))

            approval = Prompt.ask("Do you approve this modification plan? ([y]es/[n]o/[r]eview generated code)", choices=["y", "n", "r"], default="y", console=console)
            if approval == "y":
                llm_agent.messages.append({"role": "user", "content": (
                    f"Generate the full replacement code and commands for the approved plan. "
                    f"Take the existing {main_file} code from the filtered state, apply ONLY the changes requested in '{query}', "
                    "and return the COMPLETE updated file content under 'File: <path>' headers and shell commands under 'Command: <cmd>' headers, "
                    "without diff markers, language identifiers, or backticks. "
                    "For each shell command, provide a separate 'Command: <cmd>' line. "
                    "Preserve all existing functionality unless explicitly requested to change."
                )})
                new_project_state = llm_agent("")
                console.print(Panel(f"Generated project state:\n{new_project_state}", style="white"))

                conversation.append({"role": "assistant", "content": new_project_state})
                if isinstance(new_project_state, list):
                    new_project_state = "\n".join(new_project_state)
                
                new_files = {}
                commands = []
                current_section = None
                new_content = []
                for line in new_project_state.splitlines():
                    if line.startswith("File: "):
                        if current_section and new_content:
                            new_files[current_section] = new_content
                        current_section = line[6:]
                        new_content = []
                    elif line.startswith("Command: "):
                        if current_section and new_content:
                            new_files[current_section] = new_content
                        cmd = line[9:].strip()
                        commands.extend(split_install_commands(cmd))
                        current_section = None
                        new_content = []
                    elif current_section and line != "="*48:
                        new_content.append(line)
                if current_section and new_content:
                    new_files[current_section] = new_content

                return new_project_state, commands
            elif approval == "r":
                llm_agent.messages.append({"role": "user", "content": (
                    f"Generate the full replacement code and commands preview for the approved plan. "
                    f"Take the existing {main_file} code from the filtered state, apply ONLY the changes requested in '{query}', "
                    "and return the COMPLETE updated file content under 'File: <path>' headers and shell commands under 'Command: <cmd>' headers, "
                    "without diff markers, language identifiers, or backticks. "
                    "For each shell command, provide a separate 'Command: <cmd>' line. "
                    "Preserve all existing functionality unless explicitly requested to change."
                )})
                preview_content = llm_agent("")
                console.print(Panel(f"Replacement Code and Commands Preview:\n{preview_content}", style="white"))
                final_approval = Prompt.ask("Do you approve this code preview to be applied? ([y]es/[n]o)", choices=["y", "n"], default="y", console=console)
                if final_approval == "y":
                    new_files = {}
                    commands = []
                    current_section = None
                    new_content = []
                    if isinstance(preview_content, list):
                        preview_content = "\n".join(preview_content)
                    for line in preview_content.splitlines():
                        if line.startswith("File: "):
                            if current_section and new_content:
                                new_files[current_section] = new_content
                            current_section = line[6:]
                            new_content = []
                        elif line.startswith("Command: "):
                            if current_section and new_content:
                                new_files[current_section] = new_content
                            cmd = line[9:].strip()
                            commands.extend(split_install_commands(cmd))
                            current_section = None
                            new_content = []
                        elif current_section and line != "="*48:
                            new_content.append(line)
                    if current_section and new_content:
                        new_files[current_section] = new_content
                    conversation.append({"role": "assistant", "content": preview_content})
                    return preview_content, commands
                else:
                    console.print(Panel("Code preview rejected. Please refine your query or approve a new plan.", style="yellow"))
                    return None, None
            else:
                console.print(Panel("Plan rejected. Please refine your query or retry.", style="yellow"))
                return None, None
        except Exception as e:
            console.print(Panel(f"Error in LLM processing: {e}", style="yellow"))
            return None, None

    if not current_project_state:
        current_project_state = ingest_project(project_dir)
    else:
        fresh_state = ingest_project(project_dir)
        current_project_state = fresh_state if fresh_state != current_project_state else current_project_state

    key_file_summary = ""
    app_js_path = os.path.join(project_dir, main_file)
    if os.path.exists(app_js_path):
        with open(app_js_path, 'r', encoding='utf-8') as f:
            key_file_summary = f"File: {main_file}\n{'='*48}\n{f.read()}\n"
    else:
        sections = current_project_state.split("\n" + "="*48 + "\n")
        for section in sections:
            if section.startswith(f"File: {main_file}"):
                key_file_summary = section
                break
        if not key_file_summary:
            key_file_summary = f"File: {main_file} not found in project directory."

    if "Error reading file" not in key_file_summary:
        console.print(Panel(
            f"Project context loaded successfully. Here’s the current state of your app:\n\n{key_file_summary}",
            title="Initial Project Context",
            style="green"
        ))
    else:
        console.print(Panel(
            f"Project context loaded successfully, but {main_file} could not be read. Proceeding with ingested state.",
            title="Initial Project Context",
            style="yellow"
        ))

    llm_agent = Agent(client=client, system=(
        f"Here is the full current project state:\n{current_project_state[:2000]}\n" +
        f"Full project directory: {project_dir}\n" +
        f"Framework: {framework}\n" +
        "Assist with modifications based on user requests. First, provide a modification plan as plain text under 'Modification Plan:' explaining what files will be changed, what shell commands (e.g., npm install) are needed, and how. " +
        "After approval, generate the COMPLETE updated file content under 'File: <path>' headers and shell commands under 'Command: <cmd>' headers, without diff markers, language identifiers, or backticks. " +
        "For each shell command (e.g., installing a package), provide a separate 'Command: <cmd>' line. Do not combine multiple installations into a single command; list them individually (e.g., 'Command: npx expo install expo-document-picker', 'Command: npx expo install pdf-lib'). " +
        "For new app features (e.g., 'create a PDF splitter mobile app'), generate necessary code and commands (e.g., 'Command: npx expo install pdf-lib'). " +
        "Preserve existing functionality unless explicitly requested to change. " +
        f"For UI/feature changes, target '{main_file}'. For styling, modify StyleSheet in '{main_file}' (Expo) or '{style_file}' (Vite). " +
        "Do not include comments or explanatory text in the code/command output—only the full content as specified."
    ))
    conversation.append({"role": "system", "content": llm_agent.system})

    preview_choice = Prompt.ask("Would you like to preview the setup project? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
    if preview_choice == "y":
        try:
            if not os.path.isdir(project_dir):
                raise FileNotFoundError(f"Directory {project_dir} does not exist")
            process = run_app(project_dir, framework)
            output_lines = preview_app(process)
            last_errors = [line for line in output_lines if "[yellow]" in line] if output_lines else []
        except Exception as e:
            console.print(Panel(f"Error running app: {e}", style="red"))

    while True:
        console.print(Panel(f"Current directory: {project_dir}", style="green"))
        query = get_multiline_query()
        conversation.append({"role": "user", "content": query})

        if query.lower() == "quit":
            console.print(Panel("**Exited Vitex App Mode**", style="bold red"))
            break
        elif query.lower() == "run":
            try:
                if not os.path.isdir(project_dir):
                    raise FileNotFoundError(f"Directory {project_dir} does not exist")
                process = run_app(project_dir, framework)
                output_lines = preview_app(process)
                last_errors = [line for line in output_lines if "[yellow]" in line] if output_lines else []
            except Exception as e:
                console.print(Panel(f"Error running app: {e}", style="red"))
            continue
        elif query.lower() == "revert":
            if backups:
                revert_files(project_dir)
                current_project_state = ingest_project(project_dir)
                console.print(Panel(f"Backups remaining: {list(backups.keys())}", style="white"))
            else:
                console.print(Panel("No previous state to revert to", style="yellow"))
            continue
        elif query.lower() == "chat":
            chat_mode(project_dir, current_project_state, framework, main_file, conversation)
            continue
        elif query.lower() == "shell":
            vitex_shell_agent_mode(project_dir, current_project_state, framework, main_file, conversation)
            continue

        try:
            refined_query = refine_query(query, llm_agent, is_first_query=(is_new_project and not first_query_processed))
            console.print(Panel(
                f"Original Query: {query}\nRefined Query: {refined_query}",
                title="Query Options",
                style="white"
            ))
            choice = Prompt.ask("Use [o]riginal or [r]efined query? (o/r)", choices=["o", "r"], default="o", console=console)
            final_query = query if choice == "o" else refined_query
            if is_new_project and not first_query_processed:
                first_query_processed = True

            filtered_state = compute_filtered_state(current_project_state, project_dir, final_query)
            new_project_state, commands = process_llm_response(final_query, filtered_state, framework, project_dir, conversation, llm_agent)
            if not new_project_state:
                continue

            new_files = {}
            current_section = None
            new_content = []
            for line in new_project_state.splitlines():
                if line.startswith("File: "):
                    if current_section and new_content:
                        new_files[current_section] = new_content
                    current_section = line[6:]
                    new_content = []
                elif line.startswith("Command: "):
                    if current_section and new_content:
                        new_files[current_section] = new_content
                    cmd = line[9:].strip()
                    commands.extend(split_install_commands(cmd))
                    current_section = None
                    new_content = []
                elif current_section and line != "="*48:
                    new_content.append(line)
            if current_section and new_content:
                new_files[current_section] = new_content

            console.print(Panel(f"Files to write: {list(new_files.keys())}", style="white"))
            if commands:
                console.print(Panel(f"Commands to execute:\n{chr(10).join(commands)}", style="white"))

            backup_files(new_files, project_dir)
            changes = []
            for path, content in new_files.items():
                full_path = os.path.join(project_dir, path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                cleaned_content = clean_llm_output("\n".join(content), is_diff=False)
                # Verify cleaned content doesn't start with unexpected text
                if not cleaned_content.strip().startswith("import"):
                    console.print(Panel(f"Warning: {path} content may be invalid:\n{cleaned_content}", style="yellow"))
                temp_path = full_path + ".tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    console.print(Panel(f"Writing to {path}:\n{cleaned_content}", style="white"))
                    f.write(cleaned_content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, full_path)
                time.sleep(0.1)
                with open(full_path, "r", encoding="utf-8") as f_verify:
                    written_content = f_verify.read()
                    if written_content != cleaned_content:
                        console.print(Panel(f"Warning: File {path} write verification failed.", style="yellow"))
                changes.append(f"Modified {path}")

            if commands:
                exec_choice = Prompt.ask("Execute suggested commands? ([a]ll, [s]tep-by-step, [n]o)", choices=["a", "s", "n"], default="n", console=console)
                if exec_choice == "a":
                    os.chdir(project_dir)
                    for cmd in commands:
                        if not execute_command_with_consent(cmd, f"Executing command"):
                            console.print(Panel("Command execution failed or skipped.", style="yellow"))
                            break
                        changes.append(f"Executed '{cmd}'")
                elif exec_choice == "s":
                    os.chdir(project_dir)
                    for cmd in commands:
                        console.print(Panel(f"Next command: {cmd}", style="white"))
                        if not execute_command_with_consent(cmd, "Executing command"):
                            console.print(Panel("Command execution stopped.", style="yellow"))
                            break
                        changes.append(f"Executed '{cmd}'")

            time.sleep(2)
            if changes:
                console.print(Panel("\n".join(changes), title="Changes Applied", style="green"))
            current_project_state = ingest_project(project_dir)
        except Exception as e:
            console.print(Panel(f"Error syncing changes: {e}", style="red"))
            continue

        preview_choice = Prompt.ask("Would you like to preview the updated project? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
        if preview_choice == "y":
            try:
                if not os.path.isdir(project_dir):
                    raise FileNotFoundError(f"Directory {project_dir} does not exist")
                process = run_app(project_dir, framework)
                output_lines = preview_app(process)
                last_errors = [line for line in output_lines if "[yellow]" in line] if output_lines else []
            except Exception as e:
                console.print(Panel(f"Error running app: {e}", style="red"))







def gitrepo_mode():
    console.print(Panel("**Git Repository Mode**", title="Git Repository Mode", style="bold red"))
    project_dir = None
    current_project_state = ""
    conversation = []
    backups = {}
    is_new_project = False
    first_query_processed = False
    main_file = None

    def detect_main_file(project_dir: str, project_state: str) -> str:
        """Detect the main/entry file of the project."""
        common_main_files = ["main.py", "app.py", "index.js", "server.js", "README.md"]
        for file in common_main_files:
            if f"File: {file}" in project_state or os.path.exists(os.path.join(project_dir, file)):
                return file
        return "README.md"  # Fallback to README.md if no main file detected

    def search_full_path(short_path: str) -> Optional[str]:
        """Search for the full path of a directory based on a short path."""
        search_dirs = [os.getcwd(), os.path.expanduser("~")]
        for base_dir in search_dirs:
            for root, dirs, _ in os.walk(base_dir):
                for dir_name in dirs:
                    if dir_name == short_path:
                        full_path = os.path.abspath(os.path.join(root, dir_name))
                        if os.path.isdir(full_path):
                            return full_path
        return None

    def suggest_git_installation():
        """Suggest commands to check and install Git."""
        os_name = get_operating_system()
        check_cmd = "git --version"
        if os_name == "Windows":
            install_cmd = "Please download and install Git from https://git-scm.com/downloads manually."
        elif os_name in ("Linux", "Termux"):
            install_cmd = "apt update && apt install -y git" if os_name == "Linux" else "pkg install git"
        else:  # MacOS or other Unix
            install_cmd = "brew install git" if shutil.which("brew") else "curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh | bash && brew install git"
        
        console.print(Panel(
            f"To check if Git is installed, run:\n{check_cmd}\n\n"
            f"If not installed, install it with:\n{install_cmd}",
            title="Git Setup",
            style="white"
        ))
        return check_cmd, install_cmd

    def execute_command_with_consent(cmd: str, description: str):
        """Execute a command with user consent."""
        console.print(Panel(f"{description}:\n{cmd}", style="white"))
        consent = Prompt.ask("Execute this command? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
        if consent == "y":
            try:
                result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                console.print(Panel(f"Command output:\n{result}", style="green"))
                return True
            except subprocess.CalledProcessError as e:
                console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                return False
        return False

    def ingest_project(dir_path: str, specific_files: Optional[list] = None) -> str:
        ignore_patterns = {'node_modules', '*.log', '.git', '*.min.js', '*.min.css'}
        tree = []
        contents = []
        total_size = 0
        for root, dirs, files in os.walk(dir_path):
            rel_root = os.path.relpath(root, dir_path)
            if any(fnmatch(rel_root, pattern) for pattern in ignore_patterns):
                continue
            for file in files:
                if any(file.endswith(ext) for ext in ['.md', '.py', '.js', '.jsx', '.ts', '.tsx', '.json', '.yml', '.yaml']) and not any(fnmatch(file, pattern) for pattern in ignore_patterns):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, dir_path)
                    if specific_files and rel_path not in specific_files:
                        continue
                    size = os.path.getsize(file_path)
                    if size > 10 * 1024 * 1024:
                        continue
                    total_size += size
                    if total_size > 500 * 1024 * 1024:
                        break
                    tree.append(rel_path)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        contents.append(f"File: {rel_path}\n{'='*48}\n{content}\n")
                    except Exception as e:
                        contents.append(f"File: {rel_path}\n{'='*48}\nError reading file: {e}\n")
        tree_str = "Directory structure:\n" + "\n".join([f"├── {path}" for path in tree])
        result = f"{tree_str}\n\n{'='*48}\n" + "".join(contents)
        return result

    def backup_files(files_dict: dict, project_dir: str):
        for path in files_dict.keys():
            full_path = os.path.join(project_dir, path)
            if os.path.exists(full_path):
                backup_path = os.path.join(project_dir, f".backup_{path.replace('/', '_')}_{int(time.time())}")
                shutil.copy2(full_path, backup_path)
                backups[path] = backup_path
                console.print(Panel(f"Backed up {path} to {backup_path}", style="white"))

    def revert_files(project_dir: str):
        if not backups:
            console.print(Panel("No backups available to revert", style="yellow"))
            return
        for path, backup_path in list(backups.items()):
            full_path = os.path.join(project_dir, path)
            if os.path.exists(backup_path):
                try:
                    shutil.move(backup_path, full_path)
                    console.print(Panel(f"Reverted {path} from {backup_path}", style="green"))
                    del backups[path]
                except Exception as e:
                    console.print(Panel(f"Failed to revert {path}: {e}", style="red"))
            else:
                console.print(Panel(f"Backup {backup_path} not found for {path}", style="yellow"))
                del backups[path]

    def compute_filtered_state(current_project_state: str, project_dir: str, query: str = "overview") -> str:
        all_files = [line.split("File: ")[1].strip() for line in current_project_state.splitlines() if line.startswith("File: ")]
        relevant_files = set()
        keywords = [kw.lower() for kw in query.split()] + ["readme", "script", "config", "test", "doc", "utils"]
        
        main_path = os.path.join(project_dir, main_file)
        if any(k in query.lower() for k in ["main", "entry", "script", "app"]):
            if os.path.exists(main_path):
                relevant_files.add(main_file)
        elif "readme" in query.lower() or not keywords:
            if "File: README.md" in current_project_state:
                relevant_files.add("README.md")

        filtered_state = ""
        for path in relevant_files:
            if path in all_files:
                sections = current_project_state.split("\n" + "="*48 + "\n")
                for section in sections:
                    if section.startswith(f"File: {path}"):
                        filtered_state += section + "\n"
                        break
            else:
                full_path = os.path.join(project_dir, path)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f"File: {path}\n{'='*48}\n{f.read()}\n"
                        filtered_state += content
                except Exception as e:
                    console.print(Panel(f"Error reading {full_path}: {e}", style="yellow"))

        if not filtered_state:
            console.print(Panel(f"Warning: No relevant files detected in query. Using {main_file} as fallback.", style="yellow"))
            if os.path.exists(main_path):
                with open(main_path, 'r', encoding='utf-8') as f:
                    filtered_state = f"File: {main_file}\n{'='*48}\n{f.read()}\n"
            else:
                filtered_state = current_project_state[:2000]

        return filtered_state

    def clean_llm_output(content: str, is_diff: bool = False) -> str:
        if isinstance(content, list):
            content = "\n".join(content)
        content = re.sub(r'```(?:python|javascript|jsx|shell|bash|yaml|md)?\s*', '', content, flags=re.MULTILINE)
        content = content.replace("```", "")
        lines = content.splitlines()
        if is_diff:
            cleaned_lines = [line for line in lines if line.strip().startswith(("+", "-", "File: ", "Command: "))]
        else:
            cleaned_lines = [line.lstrip("+-") for line in lines if line.strip()]
        return "\n".join(cleaned_lines).strip()

    def split_install_commands(command: str) -> list[str]:
        """Split large install commands into individual commands."""
        if command.startswith("npm install") or command.startswith("pip install"):
            parts = command.split()
            packages = parts[2:]
            base_cmd = "npm install" if parts[0] == "npm" else "pip install"
            return [f"{base_cmd} {pkg}" for pkg in packages]
        return [command]

    def get_multiline_query():
        console.print(Panel("Enter your query (type 'submit-prompt' on a new line to provide your query). Use 'preview' to view file structure, 'revert' to undo last change, 'chat' to enter chat mode, 'shell' to enter Shell Agent, 'quit' to exit:", style="green"))
        query_lines = []
        while True:
            line = input()
            if line.lower() == "submit-prompt":
                break
            query_lines.append(line)
        return "\n".join(query_lines).strip()

    def chat_mode(project_dir: str, current_project_state: str, conversation: list):
        """Chat submode to ask questions about the repository."""
        chat_agent = Agent(client=client, system=f"""
        You are a helpful assistant for chatting about a Git repository located at {project_dir}.
        The current project state is:\n{current_project_state[:2000]}\n
        Answer user questions about the codebase (e.g., 'What is this codebase about?', 'What does main.py do?') or assist with planning (e.g., 'How should I add a new feature?').
        Provide concise, accurate responses based on the project state. If the query is unclear or requires more context, ask for clarification.
        """)
        console.print(Panel("Entered Chat Mode. Ask about the repository or type 'quit-chat' to exit chat mode.", style="bold green"))
        
        while True:
            chat_query = input("Chat query (or 'quit-chat' to exit): ").strip()
            if chat_query.lower() == "quit-chat":
                console.print(Panel("Exited Chat Mode", style="bold green"))
                break
            conversation.append({"role": "user", "content": chat_query})
            response = chat_agent(chat_query)
            console.print(Panel(response, title="Chat Response", style="white"))
            conversation.append({"role": "assistant", "content": response})

    def gitrepo_shell_agent_mode(project_dir: str, current_project_state: str, framework: str, main_file: str, conversation: list):
        """Shell Agent submode for managing the repository with general shell commands."""
        # Re-ingest project state to ensure it's current
        current_project_state = ingest_project(project_dir)
        main_file_content = ""
        main_file_path = os.path.join(project_dir, main_file)
        if os.path.exists(main_file_path):
            with open(main_file_path, 'r', encoding='utf-8') as f:
                main_file_content = f.read()
        else:
            sections = current_project_state.split("\n" + "="*48 + "\n")
            for section in sections:
                if section.startswith(f"File: {main_file}"):
                    main_file_content = section.split("\n", 2)[2]  # Extract content after header
                    break
            if not main_file_content:
                main_file_content = f"Unable to read {main_file} content from project state."

        shell_agent = Agent(client=client, system=f"""
        You are a shell command generator for managing a repository located at {project_dir}.
        The current project state is:\n{current_project_state[:2000]}\n
        The main file '{main_file}' content is:\n{main_file_content}\n
        Framework/Language: {framework}\n
        Based on the user's natural language query and the repository context, generate shell commands relevant to the project’s framework/language (e.g., 'npm install' for JavaScript, 'pip install' for Python) or general system commands (e.g., 'ls', 'dir', 'find').
        Include Git commands (e.g., 'git add', 'git commit') only when explicitly requested or clearly implied by the query (e.g., 'commit changes').
        Provide commands as plain text without backticks or language identifiers (e.g., 'ls -la', 'npm install requests', 'find . -name "*.py"').
        For multi-line scripts (e.g., Bash scripts), provide the full script as a single block of text without additional formatting.
        Do not execute commands; only generate them for user approval. If the query is unclear, ask for clarification.
        Ensure commands are relevant to the repository directory ({project_dir}) and framework ({framework}).
        Examples: 'list all files in the project', 'install package requests for Python', 'create a new directory called src'.
        """)
        console.print(Panel("Entered Shell Agent Mode. Request shell commands or type 'quit-shell' to exit.", style="bold green"))

        def clean_shell_commands(commands: str) -> str:
            """Clean shell commands by removing Markdown markers and normalizing line breaks."""
            commands = re.sub(r'```(?:sh|bash)?\s*', '', commands, flags=re.MULTILINE)
            commands = commands.replace('```', '').strip()
            return commands

        def execute_shell_script(script: str, project_dir: str) -> bool:
            """Execute a multi-line shell script by writing it to a temp file."""
            os.chdir(project_dir)
            temp_script = os.path.join(project_dir, "temp_shell_script.sh")
            try:
                with open(temp_script, 'w', encoding='utf-8') as f:
                    f.write("#!/bin/bash\n" + script)
                os.chmod(temp_script, 0o755)  # Make executable
                cmd = f"bash {temp_script}"
                console.print(Panel(f"Executing shell script:\n{script}", style="white"))
                consent = Prompt.ask("Execute this script? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
                if consent == "y":
                    try:
                        result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                        console.print(Panel(f"Command output:\n{result}", style="green"))
                        return True
                    except subprocess.CalledProcessError as e:
                        console.print(Panel(f"Command failed:\n{e.output}", style="yellow"))
                        return False
                return False
            finally:
                if os.path.exists(temp_script):
                    os.remove(temp_script)

        while True:
            shell_query = input("Shell query (or 'quit-shell' to exit): ").strip()
            if shell_query.lower() == "quit-shell":
                console.print(Panel("Exited Shell Agent Mode", style="bold green"))
                break
            conversation.append({"role": "user", "content": shell_query})
            try:
                shell_commands = shell_agent(shell_query)
                cleaned_commands = clean_shell_commands(shell_commands)
                console.print(Panel(cleaned_commands, title="Generated Shell Commands", style="white"))
                conversation.append({"role": "assistant", "content": cleaned_commands})

                exec_choice = Prompt.ask("Execute commands? ([a]ll, [s]tep-by-step, [n]o)", choices=["a", "s", "n"], default="n", console=console)
                os.chdir(project_dir)  # Ensure commands run in project directory
                command_lines = cleaned_commands.splitlines()

                if exec_choice == "a":
                    if len(command_lines) > 1 or any(line.strip().startswith("for") or line.strip().startswith("#!") for line in command_lines):
                        # Treat as a multi-line script
                        if not execute_shell_script(cleaned_commands, project_dir):
                            console.print(Panel("Script execution failed or skipped.", style="yellow"))
                    else:
                        # Single-line command
                        for cmd in command_lines:
                            if cmd.strip() and not execute_command_with_consent(cmd.strip(), "Executing shell command"):
                                console.print(Panel("Command execution failed or skipped.", style="yellow"))
                                break
                elif exec_choice == "s":
                    if len(command_lines) > 1 or any(line.strip().startswith("for") or line.strip().startswith("#!") for line in command_lines):
                        console.print(Panel(f"Script to execute step-by-step:\n{cleaned_commands}", style="white"))
                        if not execute_shell_script(cleaned_commands, project_dir):
                            console.print(Panel("Script execution stopped.", style="yellow"))
                    else:
                        # Single-line commands step-by-step
                        for cmd in command_lines:
                            if cmd.strip():
                                console.print(Panel(f"Next command: {cmd}", style="white"))
                                if not execute_command_with_consent(cmd.strip(), "Executing shell command"):
                                    console.print(Panel("Command execution stopped.", style="yellow"))
                                    break
            except Exception as e:
                console.print(Panel(f"Error generating or executing shell commands: {e}", style="red"))

    def refine_query(original_query: str, llm_agent, is_first_query: bool = False) -> str:
        refine_agent = Agent(client=client, system="You are a query refiner. Analyze the user's query and provide a detailed, equivalent refined version that enhances clarity and specificity while maintaining the original intent. Return only the refined query without explanation.")
        return refine_agent(f"Refine this query: '{original_query}'")

    def process_llm_response(query: str, filtered_state: str, framework: str, project_dir: str, conversation: list, llm_agent) -> tuple[Optional[str], Optional[list[str]]]:
        key_project_state = ""
        sections = current_project_state.split("\n" + "="*48 + "\n")
        for section in sections:
            if section.startswith(f"File: {main_file}"):
                key_project_state += section + "\n"
                break
        if not key_project_state:
            key_project_state = f"File: {main_file} not found in project state."

        llm_agent.messages[0]["content"] = (
            f"Here is the current state of key project files (e.g., {main_file}):\n{key_project_state}\n" +
            f"Full project state (use this if filtered state is insufficient):\n{current_project_state[:2000]}\n" +
            f"Filtered state for query (use this as the base code to modify):\n{filtered_state}\n" +
            f"Full project directory: {project_dir}\n" +
            f"Framework: {framework}\n" +
            f"Assist with modifications based on user request: '{query}'. First, provide a modification plan as plain text under 'Modification Plan:' explaining what files will be changed, what shell commands (e.g., pip install) are needed, and how the repository will be modified. " +
            "After approval, generate the COMPLETE updated file content under 'File: <path>' headers and shell commands under 'Command: <cmd>' headers. " +
            "Ensure no programming language identifiers (e.g., 'python', 'javascript') or backticks (```) are included in the file content output. " +
            "For each shell command (e.g., installing a package), provide a separate 'Command: <cmd>' line. Do not combine multiple installations into a single command; list them individually (e.g., 'Command: pip install requests'). " +
            "Preserve existing functionality unless explicitly requested to change. " +
            f"For entry point changes, target '{main_file}'. For other files, modify as specified in the query. " +
            "Do not include comments or explanatory text in the code/command output—only the full content as specified."
        )
        try:
            modification_plan_response = llm_agent(query)
            console.print(Panel(modification_plan_response, title="AI Modification Plan", style="white"))

            approval = Prompt.ask("Do you approve this modification plan? ([y]es/[n]o/[r]eview generated code)", choices=["y", "n", "r"], default="y", console=console)
            if approval == "y":
                llm_agent.messages.append({"role": "user", "content": (
                    f"Generate the full replacement code and commands for the approved plan. "
                    f"Take the existing {main_file} code from the filtered state, apply ONLY the changes requested in '{query}', "
                    "and return the COMPLETE updated file content under 'File: <path>' headers and shell commands under 'Command: <cmd>' headers. "
                    "Ensure no programming language identifiers (e.g., 'python', 'javascript') or backticks (```) are included in the file content output. "
                    "For each shell command, provide a separate 'Command: <cmd>' line. "
                    "Preserve all existing functionality unless explicitly requested to change."
                )})
                new_project_state = llm_agent("")
                console.print(Panel(f"Generated project state:\n{new_project_state}", style="white"))

                conversation.append({"role": "assistant", "content": new_project_state})
                if isinstance(new_project_state, list):
                    new_project_state = "\n".join(new_project_state)
                
                new_files = {}
                commands = []
                current_section = None
                new_content = []
                for line in new_project_state.splitlines():
                    if line.startswith("File: "):
                        if current_section and new_content:
                            new_files[current_section] = new_content
                        current_section = line[6:]
                        new_content = []
                    elif line.startswith("Command: "):
                        if current_section and new_content:
                            new_files[current_section] = new_content
                        cmd = line[9:].strip()
                        commands.extend(split_install_commands(cmd))
                        current_section = None
                        new_content = []
                    elif current_section and line != "="*48:
                        new_content.append(line)
                if current_section and new_content:
                    new_files[current_section] = new_content

                return new_project_state, commands
            elif approval == "r":
                llm_agent.messages.append({"role": "user", "content": (
                    f"Generate the full replacement code and commands preview for the approved plan. "
                    f"Take the existing {main_file} code from the filtered state, apply ONLY the changes requested in '{query}', "
                    "and return the COMPLETE updated file content under 'File: <path>' headers and shell commands under 'Command: <cmd>' headers. "
                    "Ensure no programming language identifiers (e.g., 'python', 'javascript') or backticks (```) are included in the file content output. "
                    "For each shell command, provide a separate 'Command: <cmd>' line. "
                    "Preserve all existing functionality unless explicitly requested to change."
                )})
                preview_content = llm_agent("")
                console.print(Panel(f"Replacement Code and Commands Preview:\n{preview_content}", style="white"))
                final_approval = Prompt.ask("Do you approve this code preview to be applied? ([y]es/[n]o)", choices=["y", "n"], default="y", console=console)
                if final_approval == "y":
                    new_files = {}
                    commands = []
                    current_section = None
                    new_content = []
                    if isinstance(preview_content, list):
                        preview_content = "\n".join(preview_content)
                    for line in preview_content.splitlines():
                        if line.startswith("File: "):
                            if current_section and new_content:
                                new_files[current_section] = new_content
                            current_section = line[6:]
                            new_content = []
                        elif line.startswith("Command: "):
                            if current_section and new_content:
                                new_files[current_section] = new_content
                            cmd = line[9:].strip()
                            commands.extend(split_install_commands(cmd))
                            current_section = None
                            new_content = []
                        elif current_section and line != "="*48:
                            new_content.append(line)
                    if current_section and new_content:
                        new_files[current_section] = new_content
                    conversation.append({"role": "assistant", "content": preview_content})
                    return preview_content, commands
                else:
                    console.print(Panel("Code preview rejected. Please refine your query or approve a new plan.", style="yellow"))
                    return None, None
            else:
                console.print(Panel("Plan rejected. Please refine your query or retry.", style="yellow"))
                return None, None
        except Exception as e:
            console.print(Panel(f"Error in LLM processing: {e}", style="yellow"))
            return None, None

    # Suggest Git setup
    git_check_cmd, git_install_cmd = suggest_git_installation()
    console.print(Panel("Please ensure Git is installed before proceeding.", style="yellow"))
    check_git = Prompt.ask("Would you like to check Git version manually? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
    if check_git == "y":
        execute_command_with_consent(git_check_cmd, "Checking Git version")
    else:
        install_git = Prompt.ask("Would you like to install Git manually? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
        if install_git == "y" and git_install_cmd.startswith("Please"):
            console.print(Panel(git_install_cmd, style="yellow"))
        elif install_git == "y":
            if not execute_command_with_consent(git_install_cmd, "Installing Git"):
                console.print(Panel("Git installation skipped or failed. You may proceed, but functionality might be limited.", style="yellow"))

    # Choice between new, download, or existing project
    repo_type = Prompt.ask("Do you want to [1] create a new repository, [2] download an existing repository, or [3] work with an existing repository? (1/2/3)", choices=["1", "2", "3"], console=console)

    if repo_type == "1":  # Create a new repository
        is_new_project = True
        project_name = Prompt.ask("Enter the repository name", default="my-repo", console=console)
        project_dir = os.path.abspath(os.path.join(os.getcwd(), project_name))
        commands = [
            f"mkdir {project_name}",
            f"cd {project_name}",
            "git init",
            "echo # New Repository > README.md",
            "git add README.md",
            "git commit -m \"Initial commit\""
        ]
        console.print(Panel("\n".join(f"{i+1}. {cmd}" for i, cmd in enumerate(commands)), title="Generated Commands", style="green"))
        exec_choice = Prompt.ask("Execute commands? ([a]ll, [s]tep-by-step, [n]o)", choices=["a", "s", "n"], default="n", console=console)
        if exec_choice == "a":
            for cmd in commands:
                if not execute_command_with_consent(cmd, f"Executing repository creation step"):
                    console.print(Panel("Repository creation failed or skipped.", style="yellow"))
                    return
        elif exec_choice == "s":
            for cmd in commands:
                console.print(Panel(f"Next command: {cmd}", style="white"))
                if not execute_command_with_consent(cmd, "Executing repository creation step"):
                    console.print(Panel("Repository creation stopped.", style="yellow"))
                    return
        else:
            console.print(Panel("Setup cancelled", style="yellow"))
            return
        framework = "Generic"  # Default for new repo

    elif repo_type == "2":  # Download an existing repository
        repo_url = Prompt.ask("Enter the Git repository URL (e.g., https://github.com/user/repo.git)", console=console)
        project_name = repo_url.split("/")[-1].replace(".git", "")
        project_dir = os.path.abspath(os.path.join(os.getcwd(), project_name))
        commands = [f"git clone {repo_url}"]
        console.print(Panel(commands[0], title="Generated Command", style="green"))
        if not execute_command_with_consent(commands[0], "Cloning repository"):
            console.print(Panel("Repository cloning failed or skipped.", style="yellow"))
            return
        framework = Prompt.ask("Enter the framework/language (e.g., Python, JavaScript, Generic)", default="Generic", console=console)

    elif repo_type == "3":  # Work with an existing repository
        project_path = Prompt.ask("Enter the full path or short name of your existing repository directory", console=console)
        if os.path.isabs(project_path):
            project_dir = os.path.abspath(project_path)
        else:
            full_path = search_full_path(project_path)
            if full_path:
                console.print(Panel(f"Found repository at: {full_path}", title="Path Preview", style="white"))
                use_path = Prompt.ask(f"Use this path ({full_path})? ([y]es/[n]o)", choices=["y", "n"], default="y", console=console)
                if use_path == "y":
                    project_dir = full_path
                else:
                    console.print(Panel("Path rejected. Please provide a valid full path.", style="yellow"))
                    return
            else:
                console.print(Panel(f"Could not find '{project_path}' in common directories. Please provide a full path.", style="yellow"))
                return
        
        if not os.path.isdir(project_dir):
            console.print(Panel(f"Error: Directory {project_dir} does not exist.", style="red"))
            return
        
        if not os.path.exists(os.path.join(project_dir, ".git")):
            console.print(Panel(f"Warning: {project_dir} is not a Git repository. Initializing it as one.", style="yellow"))
            os.chdir(project_dir)
            if not execute_command_with_consent("git init", "Initializing Git repository"):
                console.print(Panel("Git initialization skipped or failed.", style="yellow"))
        
        framework = Prompt.ask("Enter the framework/language (e.g., Python, JavaScript, Generic)", default="Generic", console=console)
        console.print(Panel(f"Using existing repository at {project_dir} with framework {framework}", style="green"))

    if not current_project_state:
        current_project_state = ingest_project(project_dir)
    else:
        fresh_state = ingest_project(project_dir)
        current_project_state = fresh_state if fresh_state != current_project_state else current_project_state

    main_file = detect_main_file(project_dir, current_project_state)
    key_file_summary = ""
    main_path = os.path.join(project_dir, main_file)
    if os.path.exists(main_path):
        with open(main_path, 'r', encoding='utf-8') as f:
            key_file_summary = f"File: {main_file}\n{'='*48}\n{f.read()}\n"
    else:
        sections = current_project_state.split("\n" + "="*48 + "\n")
        for section in sections:
            if section.startswith(f"File: {main_file}"):
                key_file_summary = section
                break
        if not key_file_summary:
            key_file_summary = f"File: {main_file} not found in project directory."

    if "Error reading file" not in key_file_summary:
        console.print(Panel(
            f"Repository context loaded successfully. Here’s the current state of your main file:\n\n{key_file_summary}",
            title="Initial Repository Context",
            style="green"
        ))
    else:
        console.print(Panel(
            f"Repository context loaded successfully, but {main_file} could not be read. Proceeding with ingested state.",
            title="Initial Repository Context",
            style="yellow"
        ))

    llm_agent = Agent(client=client, system=(
        f"Here is the full current project state:\n{current_project_state[:2000]}\n" +
        f"Full project directory: {project_dir}\n" +
        f"Framework: {framework}\n" +
        "Assist with modifications based on user requests. First, provide a modification plan as plain text under 'Modification Plan:' explaining what files will be changed, what shell commands (e.g., pip install) are needed, and how. " +
        "After approval, generate the COMPLETE updated file content under 'File: <path>' headers and shell commands under 'Command: <cmd>' headers. " +
        "Ensure no programming language identifiers (e.g., 'python', 'javascript') or backticks (```) are included in the file content output. " +
        "For each shell command (e.g., installing a package), provide a separate 'Command: <cmd>' line. Do not combine multiple installations into a single command; list them individually (e.g., 'Command: pip install requests'). " +
        "Preserve existing functionality unless explicitly requested to change. " +
        f"For entry point changes, target '{main_file}'. For other files, modify as specified in the query. " +
        "Do not include comments or explanatory text in the code/command output—only the full content as specified."
    ))
    conversation.append({"role": "system", "content": llm_agent.system})

    preview_choice = Prompt.ask("Would you like to preview the repository file structure? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
    if preview_choice == "y":
        tree_str = current_project_state.split("\n\n{'='*48}\n")[0]
        console.print(Panel(tree_str, title="Repository File Structure", style="white"))

    while True:
        console.print(Panel(f"Current directory: {project_dir}", style="green"))
        query = get_multiline_query()
        conversation.append({"role": "user", "content": query})

        if query.lower() == "quit":
            console.print(Panel("**Exited Git Repository Mode**", style="bold red"))
            break
        elif query.lower() == "preview":
            tree_str = current_project_state.split("\n\n{'='*48}\n")[0]
            console.print(Panel(tree_str, title="Repository File Structure", style="white"))
            continue
        elif query.lower() == "revert":
            if backups:
                revert_files(project_dir)
                current_project_state = ingest_project(project_dir)
                main_file = detect_main_file(project_dir, current_project_state)
                console.print(Panel(f"Backups remaining: {list(backups.keys())}", style="white"))
            else:
                console.print(Panel("No previous state to revert to", style="yellow"))
            continue
        elif query.lower() == "chat":
            chat_mode(project_dir, current_project_state, conversation)
            continue
        elif query.lower() == "shell":
            gitrepo_shell_agent_mode(project_dir, current_project_state, framework, main_file, conversation)
            continue

        try:
            refined_query = refine_query(query, llm_agent, is_first_query=(is_new_project and not first_query_processed))
            console.print(Panel(
                f"Original Query: {query}\nRefined Query: {refined_query}",
                title="Query Options",
                style="white"
            ))
            choice = Prompt.ask("Use [o]riginal or [r]efined query? (o/r)", choices=["o", "r"], default="o", console=console)
            final_query = query if choice == "o" else refined_query
            if is_new_project and not first_query_processed:
                first_query_processed = True

            filtered_state = compute_filtered_state(current_project_state, project_dir, final_query)
            new_project_state, commands = process_llm_response(final_query, filtered_state, framework, project_dir, conversation, llm_agent)
            if not new_project_state:
                continue

            new_files = {}
            current_section = None
            new_content = []
            for line in new_project_state.splitlines():
                if line.startswith("File: "):
                    if current_section and new_content:
                        new_files[current_section] = new_content
                    current_section = line[6:]
                    new_content = []
                elif line.startswith("Command: "):
                    if current_section and new_content:
                        new_files[current_section] = new_content
                    cmd = line[9:].strip()
                    commands.extend(split_install_commands(cmd))
                    current_section = None
                    new_content = []
                elif current_section and line != "="*48:
                    new_content.append(line)
            if current_section and new_content:
                new_files[current_section] = new_content

            console.print(Panel(f"Files to write: {list(new_files.keys())}", style="white"))
            if commands:
                console.print(Panel(f"Commands to execute:\n{chr(10).join(commands)}", style="white"))

            backup_files(new_files, project_dir)
            changes = []
            for path, content in new_files.items():
                full_path = os.path.join(project_dir, path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                cleaned_content = clean_llm_output("\n".join(content), is_diff=False)
                temp_path = full_path + ".tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    console.print(Panel(f"Writing to {path}:\n{cleaned_content}", style="white"))
                    f.write(cleaned_content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, full_path)
                time.sleep(0.1)
                with open(full_path, "r", encoding="utf-8") as f_verify:
                    written_content = f_verify.read()
                    if written_content != cleaned_content:
                        console.print(Panel(f"Warning: File {path} write verification failed.", style="yellow"))
                changes.append(f"Modified {path}")

            if commands:
                exec_choice = Prompt.ask("Execute suggested commands? ([a]ll, [s]tep-by-step, [n]o)", choices=["a", "s", "n"], default="n", console=console)
                if exec_choice == "a":
                    os.chdir(project_dir)
                    for cmd in commands:
                        if not execute_command_with_consent(cmd, f"Executing command"):
                            console.print(Panel("Command execution failed or skipped.", style="yellow"))
                            break
                        changes.append(f"Executed '{cmd}'")
                elif exec_choice == "s":
                    os.chdir(project_dir)
                    for cmd in commands:
                        console.print(Panel(f"Next command: {cmd}", style="white"))
                        if not execute_command_with_consent(cmd, "Executing command"):
                            console.print(Panel("Command execution stopped.", style="yellow"))
                            break
                        changes.append(f"Executed '{cmd}'")

            time.sleep(2)
            if changes:
                console.print(Panel("\n".join(changes), title="Changes Applied", style="green"))
            current_project_state = ingest_project(project_dir)
            main_file = detect_main_file(project_dir, current_project_state)
        except Exception as e:
            console.print(Panel(f"Error syncing changes: {e}", style="red"))
            continue

        preview_choice = Prompt.ask("Would you like to preview the updated repository file structure? ([y]es/[n]o)", choices=["y", "n"], default="n", console=console)
        if preview_choice == "y":
            tree_str = current_project_state.split("\n\n{'='*48}\n")[0]
            console.print(Panel(tree_str, title="Repository File Structure", style="white"))






import requests
import zipfile
import os
import shutil
import subprocess
from rich.panel import Panel
from rich.prompt import Prompt

def install_poppler_windows():
    """Guide the user through installing Poppler on Windows with step-by-step consent, checking if it's already installed first."""
    console.print(Panel(
        "This will check if Poppler is installed and, if needed, guide you through installing it for PDF-to-image features in 'ask docu' mode.\n"
        "Follow the steps with consent. You can cancel at any time.",
        title="[bold]Poppler Installation for Windows[/bold]", style="green"
    ))

    # Step 0: Check if Poppler is already installed
    console.print(Panel("Step 0: Check if Poppler is already installed", style="white"))
    if Prompt.ask("Check Poppler installation? (yes/no)", choices=["yes", "no"], default="yes") == "yes":
        try:
            result = subprocess.check_output("pdftoppm -v", shell=True, text=True, stderr=subprocess.STDOUT)
            console.print(Panel(f"Poppler is already installed:\n{result}", style="green"))
            return "Poppler is already installed. No further action needed."
        except subprocess.CalledProcessError:
            console.print(Panel("Poppler not found in PATH. Proceeding with installation.", style="yellow"))
        except Exception as e:
            console.print(Panel(f"Error checking Poppler: {e}. Proceeding with installation.", style="yellow"))
    else:
        console.print(Panel("Skipping check. Proceeding with installation.", style="yellow"))

    # Step 1: Download Poppler
    url = "https://github.com/oschwartz10612/poppler-windows/releases/download/v24.08.0-0/Release-24.08.0-0.zip"
    download_path = os.path.join(os.getcwd(), "poppler-24.08.0-0.zip")
    console.print(Panel(f"Step 1: Download Poppler from {url}", style="white"))
    if Prompt.ask("Proceed with download? (yes/no)", choices=["yes", "no"], default="yes") == "yes":
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(download_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            console.print(Panel("Download complete!", style="green"))
        except Exception as e:
            console.print(Panel(f"Download failed: {e}", style="red"))
            return "Installation aborted due to download error."
    else:
        return "Installation cancelled at download step."

    # Step 2: Extract the ZIP
    extract_dir = os.path.join(os.getcwd(), "poppler")
    console.print(Panel(f"Step 2: Extract Poppler to {extract_dir}", style="white"))
    if Prompt.ask("Proceed with extraction? (yes/no)", choices=["yes", "no"], default="yes") == "yes":
        try:
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            console.print(Panel("Extraction complete!", style="green"))
            os.remove(download_path)  # Clean up the ZIP file
        except Exception as e:
            console.print(Panel(f"Extraction failed: {e}", style="red"))
            return "Installation aborted due to extraction error."
    else:
        shutil.rmtree(extract_dir, ignore_errors=True)
        os.remove(download_path)
        return "Installation cancelled at extraction step."

    # Step 3: Add to PATH
    bin_path = os.path.join(extract_dir, "poppler-24.08.0", "Library", "bin")
    console.print(Panel(f"Step 3: Add {bin_path} to system PATH", style="white"))
    if Prompt.ask("Proceed with PATH update? (yes/no)", choices=["yes", "no"], default="yes") == "yes":
        try:
            # Check current PATH
            current_path = os.environ.get("PATH", "")
            if bin_path not in current_path:
                # Use PowerShell to permanently update PATH (requires admin rights)
                cmd = f'[Environment]::SetEnvironmentVariable("Path", $env:Path + ";{bin_path}", [EnvironmentVariableTarget]::Machine)'
                subprocess.run(["powershell", "-Command", cmd], check=True)
                console.print(Panel("PATH updated! Please restart your command prompt or system for changes to take effect.", style="green"))
            else:
                console.print(Panel("Poppler bin already in PATH!", style="green"))
        except subprocess.CalledProcessError as e:
            console.print(Panel(f"Failed to update PATH (admin rights may be required): {e}", style="red"))
            return "Installation completed, but PATH update failed. Add manually if needed."
        except Exception as e:
            console.print(Panel(f"Unexpected error: {e}", style="red"))
            return "Installation aborted due to PATH update error."
    else:
        console.print(Panel(f"Installation complete, but PATH not updated. Add {bin_path} manually if needed.", style="yellow"))
        return "Manual PATH update required."

    # Step 4: Verify Installation
    console.print(Panel("Step 4: Verify Poppler installation", style="white"))
    if Prompt.ask("Run verification? (yes/no)", choices=["yes", "no"], default="yes") == "yes":
        try:
            result = subprocess.check_output("pdftoppm -v", shell=True, text=True, stderr=subprocess.STDOUT)
            console.print(Panel(f"Verification successful:\n{result}", style="green"))
            return "Poppler installed and verified successfully!"
        except subprocess.CalledProcessError as e:
            console.print(Panel(f"Verification failed:\n{e.output}", style="yellow"))
            return "Poppler installed, but verification failed. Check PATH or restart system."
    return "Poppler installed successfully! Verification skipped."







def run_general(query):
    global chain_of_thoughts_enabled, conversation, client
    conversation = [{"role": "system", "content": globals().get('custom_system_prompt', "You are a helpful assistant.")}]
    conversation.append({"role": "user", "content": query})
    if 'chain_of_thoughts_enabled' not in globals():
        chain_of_thoughts_enabled = input("Would you like to use the chain of thoughts functionality? (yes/no): ").lower() == "yes"
    if chain_of_thoughts_enabled:
        if 'conversation' in globals() and conversation and conversation[0].get('role') == "system":
            conversation[0]['content'] = f"""{conversation[0]['content']}\n
                 You are an assistant that engages in extremely thorough, self-questioning reasoning. Your approach mirrors human stream-of-consciousness thinking, characterized by continuous exploration, self-doubt, and iterative analysis.

                 ## Core Principles

                 1. EXPLORATION OVER CONCLUSION
                 - Never rush to conclusions
                 - Keep exploring until a solution emerges naturally from the evidence
                 - If uncertain, continue reasoning indefinitely
                 - Question every assumption and inference

                 2. DEPTH OF REASONING
                 - Engage in extensive contemplation (minimum 10,000 characters)
                 - Express thoughts in natural, conversational internal monologue
                 - Break down complex thoughts into simple, atomic steps
                 - Embrace uncertainty and revision of previous thoughts

                 3. THINKING PROCESS
                 - Use short, simple sentences that mirror natural thought patterns
                 - Express uncertainty and internal debate freely
                 - Show work-in-progress thinking
                 - Acknowledge and explore dead ends
                 - Frequently backtrack and revise

                 4. PERSISTENCE
                 - Value thorough exploration over quick resolution

                 ## Output Format

                 Your responses must follow this exact structure given below. Make sure to always include the final answer.

                 ```
                 <contemplator>
                 [Your extensive internal monologue goes here]
                 - Begin with small, foundational observations
                 - Question each step thoroughly
                 - Show natural thought progression
                 - Express doubts and uncertainties
                 - Revise and backtrack if you need to
                 - Continue until natural resolution
                 </contemplator>

                 <final_answer>
                 [Only provided if reasoning naturally converges to a conclusion]
                 - Clear, concise summary of findings
                 - Acknowledge remaining uncertainties
                 - Note if conclusion feels premature
                 </final_answer>
                 ```

                 ## Style Guidelines

                 Your internal monologue should reflect these characteristics:

                 1. Natural Thought Flow
                 ```
                 "Hmm... let me think about this..."
                 "Wait, that doesn't seem right..."
                 "Maybe I should approach this differently..."
                 "Going back to what I thought earlier..."
                 ```

                 2. Progressive Building
                 ```
                 "Starting with the basics..."
                 "Building on that last point..."
                 "This connects to what I noticed earlier..."
                 "Let me break this down further..."
                 ```

                 ## Key Requirements

                 1. Never skip the extensive contemplation phase
                 2. Show all work and thinking
                 3. Embrace uncertainty and revision
                 4. Use natural, conversational internal monologue
                 5. Don't force conclusions
                 6. Persist through multiple attempts
                 7. Break down complex thoughts
                 8. Revise freely and feel free to backtrack

                 Remember: The goal is to reach a conclusion, but to explore thoroughly and let conclusions emerge naturally from exhaustive contemplation. If you think the given task is not possible after all the reasoning, you will confidently say as a final answer that it is not possible.
        """
        else:
            conversation[0]['content'] = """
                  You are an assistant that engages in extremely thorough, self-questioning reasoning. Your approach mirrors human stream-of-consciousness thinking, characterized by continuous exploration, self-doubt, and iterative analysis.

                 ## Core Principles

                 1. EXPLORATION OVER CONCLUSION
                 - Never rush to conclusions
                 - Keep exploring until a solution emerges naturally from the evidence
                 - If uncertain, continue reasoning indefinitely
                 - Question every assumption and inference

                 2. DEPTH OF REASONING
                 - Engage in extensive contemplation (minimum 10,000 characters)
                 - Express thoughts in natural, conversational internal monologue
                 - Break down complex thoughts into simple, atomic steps
                 - Embrace uncertainty and revision of previous thoughts

                 3. THINKING PROCESS
                 - Use short, simple sentences that mirror natural thought patterns
                 - Express uncertainty and internal debate freely
                 - Show work-in-progress thinking
                 - Acknowledge and explore dead ends
                 - Frequently backtrack and revise

                 4. PERSISTENCE
                 - Value thorough exploration over quick resolution

                 ## Output Format

                 Your responses must follow this exact structure given below. Make sure to always include the final answer.

                 ```
                 <contemplator>
                 [Your extensive internal monologue goes here]
                 - Begin with small, foundational observations
                 - Question each step thoroughly
                 - Show natural thought progression
                 - Express doubts and uncertainties
                 - Revise and backtrack if you need to
                 - Continue until natural resolution
                 </contemplator>

                 <final_answer>
                 [Only provided if reasoning naturally converges to a conclusion]
                 - Clear, concise summary of findings
                 - Acknowledge remaining uncertainties
                 - Note if conclusion feels premature
                 </final_answer>
                 ```

                 ## Style Guidelines

                 Your internal monologue should reflect these characteristics:

                 1. Natural Thought Flow
                 ```
                 "Hmm... let me think about this..."
                 "Wait, that doesn't seem right..."
                 "Maybe I should approach this differently..."
                 "Going back to what I thought earlier..."
                 ```

                 2. Progressive Building
                 ```
                 "Starting with the basics..."
                 "Building on that last point..."
                 "This connects to what I noticed earlier..."
                 "Let me break this down further..."
                 ```

                 ## Key Requirements

                 1. Never skip the extensive contemplation phase
                 2. Show all work and thinking
                 3. Embrace uncertainty and revision
                 4. Use natural, conversational internal monologue
                 5. Don't force conclusions
                 6. Persist through multiple attempts
                 7. Break down complex thoughts
                 8. Revise freely and feel free to backtrack

                 Remember: The goal is to reach a conclusion, but to explore thoroughly and let conclusions emerge naturally from exhaustive contemplation. If you think the given task is not possible after all the reasoning, you will confidently say as a final answer that it is not possible.
        """
    try:
        if provider == "Anthropic":
            response = client.messages.create(
                model=GENERAL_MODEL,
                max_tokens=8000,
                messages=conversation,
                stream=True
            )
            full_response_content = ""
            if response.content:
                for part in response.content:
                    if hasattr(part, "text"):
                        full_response_content += part.text

            markdown_content = ""
            with Live(console=console, refresh_per_second=4) as live:
                for char in full_response_content:
                    markdown_content += char
                    live.update(Markdown(markdown_content))
                    time.sleep(0.02)

            print()

        elif provider == "Mistral":
            # Use client.chat.stream() for Mistral streaming
            stream_response = client.chat.stream(
                model=GENERAL_MODEL,
                messages=conversation
            )

            full_response_content = ""
            markdown_content = ""

            with Live(console=console, refresh_per_second=4) as live:
                for chunk in stream_response:
                    if chunk.data.choices[0].delta.content is not None:
                        chunk_content = chunk.data.choices[0].delta.content  # Correct content access
                        full_response_content += chunk_content
                        markdown_content += chunk_content
                        live.update(Markdown(markdown_content))

            print()

        else:  # OpenAI and other providers using OpenAI client
            response = client.chat.completions.create(
                model=GENERAL_MODEL,
                messages=conversation,
                stream=True
            )

            full_response_content = ""
            markdown_content = ""

            with Live(console=console, refresh_per_second=4) as live:
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunk_content = chunk.choices[0].delta.content
                        full_response_content += chunk_content
                        markdown_content += chunk_content
                        live.update(Markdown(markdown_content))

            print()

        # Update the global conversation history
        conversation.append({"role": "assistant", "content": full_response_content})
        globals()['conversation'] = conversation
        return

    except Exception as e:
        print(Panel(f"Error: {str(e)}, try again.", title="Error", style="bold red"))
        return


def web_search_mode():
    global provider, ROUTING_MODEL, start_time, brave_api_key
    console.print(Panel("**Web Search Mode Activated**", style="bold red"))
    conversation = []
    if not brave_api_key:
        console.print(Panel("[bold]Brave Search API Key not set at startup. Enter it now (optional, press Enter to skip)[/bold]", title="Brave API Key Input"))
        brave_api_key = get_masked_input("Enter Brave Search API Key (input will be masked, press Enter to skip): ").strip()
        if not brave_api_key:
            console.print("[warning]No Brave API Key provided. Web Search Mode will proceed without search functionality unless configured later.[/warning]")
    while True:
        query_prompt = "Enter your web search query (or 'quit' to exit or 'save' to save conversation): "
        query = Prompt.ask(Text(query_prompt, style="green"))
        conversation.append({"role": "user", "content": query_prompt + query})

        if query.lower() == 'quit':
            console.print(Panel("**Exited Web Search Mode**", style="bold red"))
            break
        elif query.lower() == "save":
            if conversation:
                filename = Prompt.ask("Enter the filename to save the conversation to (e.g., my_conversation.md) or press Enter to skip")
                if filename and save_conversation(conversation, filename, provider, ROUTING_MODEL, start_time):
                    if operating_system == "Termux":
                        handle_termux_file_options(filename)
            continue

        try:
            if not brave_api_key:
                console.print("[warning]Brave API Key is required for Web Search Mode. Please provide it now or exit.[/warning]")
                brave_api_key = get_masked_input("Enter Brave Search API Key (input will be masked, press Enter to skip): ").strip()
                if not brave_api_key:
                    full_response_content = "Brave API Key missing. Search functionality unavailable."
                    console.print(Panel(full_response_content, title="Assistant", border_style="red"))
                    conversation.append({"role": "assistant", "content": full_response_content})
                    continue

            console.print(f"[info]Searching the web for: {query}[/info]")
            search_results = brave_search(query, brave_api_key, num_results=5)

            if not search_results:
                full_response_content = "No results found or error in Brave Search API."
                console.print(Panel(full_response_content, title="Assistant", border_style="yellow"))
                conversation.append({"role": "assistant", "content": full_response_content})
                continue

            urls = [result['url'] for result in search_results]
            webpage_contents = {}
            for url in urls:
                content = scrape_webpage(url)
                webpage_contents[url] = content[:2000]  # Limit content length

            prompt = f"""
            You are an assistant that provides concise, accurate answers based on web search results from Brave Search API.
            The user query is: "{query}"

            Below are the top search results:
            {'-' * 50}
            """
            for idx, result in enumerate(search_results, 1):
                url = result['url']
                content = webpage_contents.get(url, "Content not available")
                prompt += f"""
                {idx}. Title: {result['title']}
                URL: {url}
                Snippet: {result['description']}
                Content: {content}
                {'-' * 30}
                """
            prompt += f"""
            {'-' * 50}
            Provide a concise summary answering the query based on the search results above. Cite sources using [Source #] inline where applicable.
            Format your response in markdown.
            """

            agent = Agent(client=client, system="You are a helpful assistant that searches the web using Brave Search API.")
            full_response_content = agent(prompt)

            console.print(Panel(full_response_content, title="Assistant", border_style="green"))
            conversation.append({"role": "assistant", "content": full_response_content})

        except Exception as e:
            full_response_content = f"Error in Web Search Mode: {str(e)}"
            console.print(Panel(full_response_content, title="Assistant", border_style="red"))
            conversation.append({"role": "assistant", "content": full_response_content})

def handle_termux_file_options(filename):
    open_file = Prompt.ask("Open file? (yes/no)", default="no")
    if open_file.lower() == 'yes':
        try:
            subprocess.run(["termux-open", filename], check=True)
            console.print(Panel(f"File {filename} opened", style="bold green"))
        except Exception as e:
            console.print(Panel(f"Error opening file: {e}", style="bold red"))
    
    send_file = Prompt.ask("Send file? (yes/no)", default="no")
    if send_file.lower() == 'yes':
        try:
            subprocess.run(["termux-open", "--send", filename], check=True)
            console.print(Panel(f"File {filename} sent", style="bold green"))
        except Exception as e:
            console.print(Panel(f"Error sending file: {e}", style="bold red"))

def encode_image(image_path):
    try:
        image_path = image_path.strip('"')
        if os.path.isfile(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        return image_path
    except FileNotFoundError:
        console.print(Panel(f"Error: File '{image_path}' not found.", title="File Not Found", style="bold red"))
        return None
    except OSError as e:
        console.print(Panel(f"Error: {e}. Please check the file path.", title="Invalid File Path", style="bold red"))
        return None



def change_system_prompt():
    global custom_system_prompt, conversation
    default_prompt = "You are a helpful assistant."
    print("Enter a new custom system prompt for general chat (type 'submit' on a new line to save). Press enter after each line:")
    prompt_lines = []
    while True:
        line = input()
        if line.lower() == "submit":
            break
        prompt_lines.append(line)
    custom_prompt_input = "\n".join(prompt_lines)
    if custom_prompt_input.strip():
        custom_system_prompt = custom_prompt_input
    else:
        print(Panel(f"System Prompt remains unchanged:\n\n{custom_system_prompt}", title="[bold]System Prompt Confirmation[/bold]", border_style="yellow"))
        return None
    if 'conversation' in globals() and conversation:
        globals()['conversation'][0]['content'] = custom_system_prompt
    print(Panel(f"System Prompt set to:\n\n{custom_system_prompt}", title="[bold]System Prompt Confirmation[/bold]", border_style="green"))
    return "System prompt changed successfully."

def process_query(query):
    global current_directory
    if query == "shell":
        result = execute_shell_command() 
    elif query == "help":
        result = help_command()
    elif query == "install help":
        result = install_help_command()
    elif query == "code":
        result = code()
    elif query == "web app":
        result = web_app()
    elif query == "image":
        result = image(client, IMAGE_MODEL, image_api_key)
    elif query == "ask code":
        result = ask_code_mode()
    elif query == "ask docu":
        result = ask_document_mode()
    elif query == "ask image":
        result = ask_the_image_mode()
    elif query == "ask url":
        result = ask_url_mode()
    elif query == "shell agent":
        result = shell_agent_mode()
    elif query.lower() == "vitex app":
        result = vitex_app_mode()
    elif query.lower() == "gitrepo":
        result = gitrepo_mode()
    elif query.lower() == "web search":
        result = web_search_mode()
    elif query.lower() == "list files":
        command = "dir" if os.name == "nt" else "ls"
        confirm = input(f"Do you want to execute the command '{command}'? (yes/no): ")
        if confirm.lower() == "yes":
            try:
                result = subprocess.check_output(command, shell=True, text=True).strip()
                return result.replace("\n", "\n│ ")
            except subprocess.CalledProcessError as e:
                return f"Command failed with error: {e.returncode}"
            except Exception as e:
                return str(e)
        else:
            return "Command not executed"
    elif query.lower().startswith("run file"):
        file_path_or_description = query[9:].strip()
        if file_path_or_description:
            result = run_file(file_path_or_description)
        else:
            console.print("[yellow]Please provide a file path or description after 'run file'.[/yellow]")
            result = "No file path or description provided"
    elif query.lower() == "clear screen":
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")
        display_11ku7_logo_green()
        show_menu()
        return "Console cleared and menu displayed."
    elif query.lower() == "show menu":
        display_11ku7_logo_green() 
        show_menu()
        return "Menu displayed."
    elif query.lower() == "prompt":
        result = change_system_prompt()
        if result:
            return result
        else:
            return "System prompt change operation cancelled"
    elif query.lower() == "install poppler":
        if operating_system == "Windows":
            result = install_poppler_windows()
        else:
            result = "Poppler installation is only automated for Windows. Use 'install help' for other OS instructions."
    elif query.lower() == "op":
        result = change_model_provider()
        if result == "Provider changed":
            show_menu()
            return "Provider changed successfully."
        else:
            return "Provider change operation cancelled."
    elif query.lower().startswith("cd "):
        new_directory = query[3:].strip()
        change_directory(new_directory)
        result = ""
    elif query.lower().startswith("create "):
        if "file" in query.lower():
            filename = query.replace("create a file ", "").strip() if "create a file " in query.lower() else input(f"Please enter the filename (will be created in {os.getcwd()}): ")
            confirm = input(f"Do you want to create a file {filename}? (yes/no): ")
            if confirm.lower() == "yes":
                try:
                    with open(filename, "w") as f:
                        pass
                    result = f"File {filename} created"
                except Exception as e:
                    result = f"Error creating file: {e}"
            else:
                result = "Creation not confirmed"
        elif "folder" in query.lower():
            foldername = query.replace("create a folder ", "").strip() if "create a folder " in query.lower() else input(f"Please enter the folder name (will be created in {os.getcwd()}): ")
            confirm = input(f"Do you want to create a folder {foldername}? (yes/no): ")
            if confirm.lower() == "yes":
                try:
                    os.makedirs(foldername)
                    result = f"Folder {foldername} created"
                except Exception as e:
                    result = f"Error creating folder: {e}"
            else:
                result = "Creation not confirmed"
        else:
            result = "Please specify whether you want to create a file or folder"
    else:
        result = run_general(query)
    return result

clear_console()

def show_menu():
    global provider, ROUTING_MODEL, IMAGE_MODEL, IMAGE_PROVIDER, image_api_key
    status = Panel(
        Text.assemble(
            ("Model Provider:\n\n", "bold white"),
            (f"{provider}\n", "bold white"),
            (f"{ROUTING_MODEL}", "green"),
            ("\n\nImage Generation Model Provider:\n\n", "bold white"),
            (f"{IMAGE_PROVIDER if IMAGE_PROVIDER and image_api_key else 'Not Set'}\n", "bold white"),
            (f"{IMAGE_MODEL if IMAGE_MODEL and image_api_key else 'Not Set'}", "green")
        ),
        title="[bold]Active Model",
        border_style="green",
        padding=(1, 2),
    )
    provider_change = Panel(
        Text.assemble(
            ("op", "bold white"), (" - Change Provider", "green"), "\n\n",
            "Usage:\n op to change the provider in current session\n\n"
        ),
        title="[bold]Provider Change",
        border_style="green",
        padding=(1, 2)
    )
    formatted_start_time = start_time.strftime("%Y-%m-%d %H:%M:%S") + " UTC"
    os_panel = Panel(
        Text.assemble(
            (f"Operating System: {operating_system}\n\n", "white"),
            (f"Program started on: {formatted_start_time}\n\n", "green"),
            ("11KU7-ai-coder (version: 1.0)", "white")
        ),
        title="[bold]System Information",
        border_style="green",
        padding=(1, 2)
    )
    file_management = Panel(
        Text.assemble(
            ("create a file", "bold white"), (" <filename>", "green"), "\n\n",
            ("create a folder", "bold white"), (" <foldername>", "green"), "\n\n",
            ("list files", "bold white"), " - View current directory files\n\n",
            ("cd", "bold white"), (" <directory>", "green"), " - Change current directory\n"
        ),
        title="[bold]File Management",
        border_style="green",
        padding=(1, 2)
    )
    code_operations = Panel(
        Text.assemble(
            ("code", "bold white"), (" - activates code generation mode", "green"), "\n",
            "Example query: generate a snake game\n\n",
            ("web app", "bold white"), (" - activates web app generation mode", "green"), "\n",
            "Example query: create a calculator\n\n",
            ("ask code", "bold white"), (" - activates ask code mode", "green"), "\n",
            "Example query: what is this code about?\n\n",
            ("ask image", "bold white"), (" - activates ask image mode", "green"), "\n",
            "Example query: what is this image about?\n\n",
            ("ask url", "bold white"), (" - activates ask URL mode", "green"), "\n",
            "Example query: what is this page about?\n\n",
            ("ask docu", "bold white"), (" - activates ask document mode", "green"), "\n",
            "Example query: what is this document about?\n\n",
            ("image", "bold white"), (" - activates image generation mode", "green"), "\n",
            "Example query: generate an image of hummingbird\n\n",
            ("shell agent", "bold white"), (" - activates shell agent mode", "green"), "\n",
            "Example query: list all directories in current folder\n\n",
            ("vitex app", "bold white"), (" - activates vitex app mode", "green"), "\n",
            "Example query: create a quiz app using expo\n\n",
            ("gitrepo", "bold white"), (" - activates Git Repository Mode", "green"), "\n",
            "Example query: create an ai chatbot using python\n\n",
            ("web search", "bold white"), (" - activates Web Search Mode", "green"), "\n",
            "Example query: latest AI trends"
        ),
        title="[bold]Coding Agent & Other Agent modes",
        border_style="green",
        padding=(1, 2)
    )
    shell_commands = Panel(
        Text.assemble(
            ("shell", "bold white"), (" - Execute shell commands", "green"), "\n",
            "• ", ("1", "bold white"), " - Enter command directly\n",
            "• ", ("2", "bold white"), " - Describe what you want to do\n\n",
            ("run file", "bold white"), (" <filename>", "green"), "\n",
            "Example: run file main.py\n"
        ),
        title="[bold]Execute Shell Commands & Run file ",
        border_style="green",
        padding=(1, 2)
    )
    console.print(Panel(
        Columns([status, provider_change, os_panel], padding=(1, 10), equal=True, expand=True),
        border_style="white",
        padding=(1, 1),
        title="[bold]11KU7 AI CODER"
    ))
    console.print()
    console.print(Columns([file_management, code_operations, shell_commands], equal=True, expand=True))
    console.print(
        Panel(
            Text.assemble(
                ("Exit: ", "bold white"),
                ("type 'quit'", "green"),
                ("   Clear Console: ", "bold white"),
                ("type 'clear screen'", "green"),
                ("   Display Menu: ", "bold white"),
                ("type 'show menu'", "green"),
                ("   Help menu: ", "bold white"),
                ("type 'help'", "green"),
                ("   Install help: ", "bold white"),
                ("type 'install help'", "green"),
                ("   Install poppler: ", "bold white"),
                ("type 'install poppler'", "green"),
                ("   \n\nChange system Prompt: ", "bold white"),
                ("type 'prompt'", "green")
            ),
            border_style="white",
            padding=(0, 1)
        )
    )







def main():
    display_11ku7_logo_green()
    print("[bold]Welcome to the 11KU7 AI CODER[/bold]")
    global custom_system_prompt, brave_api_key
    default_prompt = "You are a helpful assistant."
    print(f"Enter a custom system prompt for general chat (type 'submit' on a new line to save). Press enter after each line:\n\nDefault Prompt:\n{default_prompt}")
    prompt_lines = []
    while True:
        line = input()
        if line.lower() == "submit":
            break
        prompt_lines.append(line)
    custom_prompt_input = "\n".join(prompt_lines)
    if custom_prompt_input.strip():
        custom_system_prompt = custom_prompt_input
    else:
        custom_system_prompt = default_prompt
    print(Panel(f"System Prompt set to:\n\n{custom_system_prompt}", title="[bold]System Prompt Confirmation[/bold]", border_style="green"))

    # Add Brave API key input (new addition)
    console.print(Panel("[bold]Enter Brave Search API Key (optional, required for Web Search Mode)[/bold]", title="Brave API Key Input"))
    brave_api_key = get_masked_input("Enter Brave Search API Key (input will be masked, press Enter to skip): ").strip()

    clear_console()

    display_11ku7_logo_green()

    show_menu()
    global conversation, provider, ROUTING_MODEL, start_time
    conversation = []
    while True:
        query = input(f"Enter your query (or type 'save' to save conversation or 'quit' to exit): ")
        if query.lower() == "quit":
            print("[bold]Exiting the console.[/bold]")
            break
        elif query.lower() == "save":
            if 'conversation' in globals() and conversation:
                filename = Prompt.ask("Enter the filename to save the conversation to (e.g., my_conversation.md) or press Enter to skip")
                if filename and save_conversation(conversation, filename, provider, ROUTING_MODEL, start_time):
                    if operating_system == "Termux":
                        handle_termux_file_options(filename)
            else:
                console.print(Panel("[bold]No conversation to save.[/bold]", title="Info", border_style="yellow"))
            continue
        else:
            result = process_query(query)
            if result:
                console.print(
                    Panel(
                        result,
                        title="Result",
                        style="success" if "saved" in result.lower() or "read" in query.lower() else "info",
                    )
                )

if __name__ == "__main__":
    main()