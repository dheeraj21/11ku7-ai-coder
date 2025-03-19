import os

def generate_11ku7_cover_page():
    """Generates and displays the full 11KU7 AI Coder cover page."""

    # ASCII Art Logo
    ascii_logo = """

                     ██╗ ██╗██╗  ██╗██╗   ██╗███████╗     █████╗ ██╗
                    ███║███║██║ ██╔╝██║   ██║╚════██║    ██╔══██╗██║
                    ╚██║╚██║█████╔╝ ██║   ██║    ██╔╝    ███████║██║
                     ██║ ██║██╔═██╗ ██║   ██║   ██╔╝     ██╔══██║██║
                     ██║ ██║██║  ██╗╚██████╔╝   ██║      ██║  ██║██║
                     ╚═╝ ╚═╝╚═╝  ╚═╝ ╚═════╝    ╚═╝      ╚═╝  ╚═╝╚═╝

                        ██████╗ ██████╗ ██████╗ ███████╗██████╗
                       ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗
                       ██║     ██║   ██║██║  ██║█████╗  ██████╔╝
                       ██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗
                       ╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║
                        ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝

    """

    # ANSI escape codes for colors
    COLORS = {
        "green": "\033[32m",
        "reset": "\033[0m",
        "bold": "\033[1m", # Adding bold for titles
        "white": "\033[37m",
    }

    # Force ANSI color application - for Termux and ANSI-supporting terminals
    def colorize_text(text, color_code=""):
        return f"{color_code}{text}{COLORS['reset']}"


    REPOSITORY_TEXT = """
https://github.com/dheeraj21/11ku7-ai-coder
    """


    DISCLAIMER_TEXT = """
11KU7 AI Coder is provided as a free, open-source tool to assist you in coding.
It's offered as is, and while we strive for it to be helpful and functional,
please understand that there are no guarantees of performance or suitability
for any specific purpose.

As an open-source project, we encourage community contributions and 
improvements. Use it responsibly and be aware of the code it generates.
    """

    TERMS_TEXT = """
By using 11KU7 AI Coder, you agree to use it in a responsible and in ethical 
manner. You are accountable for the code you generate and its use.
Remember that using external AI services may have their own terms of service.
Please keep your API keys secure.
    """

    LICENSE_TEXT = """
This project is released under the **MIT License**, a permissive open-source 
license. You are free to use, modify, and distribute it.
See the full license at: https://opensource.org/licenses/MIT
    """


    RESPONSIBLE_USE_TEXT = """
Using 11KU7 AI Coder Responsibly & Powerfully:

- Ethical & Mindful Coding: Always use 11KU7 AI Coder for ethical purposes. 
  Refrain from generating code for harmful activities or copyright 
  infringement.
- Review & Understand Code: Treat AI-generated code as a starting point. 
  Thoroughly review, understand, and test all code before deployment. 
  AI assists, but human oversight is crucial.
- API Key Security & Provider Awareness: Your API keys are your responsibility; 
  keep them secure. Be aware of the terms of service and usage policies of your 
  chosen AI model providers. 11KU7 AI Coder enhances security by masking API key 
  inputs, but ultimate security rests with you.
- Embrace Local Models for Privacy & Control: 11KU7 AI Coder champions local 
  models via Ollama and llama.cpp. Experience the power of running models 
  directly on your machine, offering enhanced privacy and control over
  your data and computations. As local models become even more powerful, 
  11KU7 AI Coder is poised to become an invincible coding companion, 
  fully independent and robust.
- Harness the Power of Prompts:
    - Custom System Prompts:  Shape the AI's behavior precisely 
      with custom system prompts.
      Tailor the assistant's role and style to perfectly match 
      your task requirements.
    - Chain of Thoughts Reasoning:  Engage the chain of thoughts 
      mode for complex tasks.
      Witness the AI's detailed, step-by-step reasoning process, 
      leading to more insightful and robust solutions.
- Provider Freedom & Flexibility:  Enjoy unparalleled freedom with 
  11KU7 AI Coder's extensive provider support. Seamlessly switch between 
  providers like Gemini, Mistral, Deepseek, Huggingface, Grok, OpenAI, Anthropic,
  Groq etc. And local options like Ollama and llama.cpp even within the same 
  session, to leverage the best model for each task.
- Explore Diverse Modes - Code Without Limits:** 11KU7 AI Coder is a toolkit 
  of possibilities. Dive into modes like:
    - Vitex/Git Repo Modes:  Transform your app development and repository 
      management workflows.
    - Web Search Mode:  Instantly access and integrate real-time web information
      into your coding process.
    - Code & Web App Generation:  Rapidly prototype, experiment, and build 
      applications with unprecedented speed.
    - Document & Code Analysis:  Unlock deep understanding from your project 
      files, documentation, and external resources, turning information into 
      actionable knowledge.
- Maximize Your Coding Workflow:
  Let 11KU7 AI Coder become your coding ally. Automate repetitive tasks, 
  accelerate development cycles, and elevate your coding efficiency to 
  new heights.
    """

    def clear_console_internal():
        os.system('cls' if os.name == 'nt' else 'clear')

    def get_color_choices_internal():
        """Hardcoded color choice - always green."""
        return ["green", "green"]

    def display_logo_hue_internal(color_names):
        """Displays the ASCII logo in green."""
        green_code = COLORS.get(color_names[0], COLORS["green"])

        colored_logo = ""
        lines = ascii_logo.splitlines()
        for line in lines:
            colored_logo += green_code + line + COLORS["reset"] + "\n"
        return colored_logo

    clear_console_internal()
    chosen_colors = get_color_choices_internal()
    logo_colored = display_logo_hue_internal(chosen_colors)

    cover_page = f"""{logo_colored}

                                    {colorize_text("(version", COLORS["white"] + COLORS["bold"])} {colorize_text("1", COLORS["white"] + COLORS["bold"])}{colorize_text(".", COLORS["white"] + COLORS["bold"])}{colorize_text("1)", COLORS["white"] + COLORS["bold"])}

                      {colorize_text("Intelligent Assistance, Right in Your Terminal", COLORS["white"])}



{colorize_text("Release Date: 12-03", COLORS["white"])}{colorize_text("-2025", COLORS["white"])}

{colorize_text("Current Version Release Date: 19-03", COLORS["white"])}{colorize_text("-2025", COLORS["white"])}

{colorize_text("Country of Origin: India", COLORS["white"])}


{colorize_text("Official Repository:", COLORS["white"] + COLORS["bold"])}{colorize_text(REPOSITORY_TEXT, COLORS["white"])}

{colorize_text("Disclaimer:", COLORS["white"] + COLORS["bold"])}{colorize_text(DISCLAIMER_TEXT, COLORS["white"])}

{colorize_text("Terms & Conditions:", COLORS["white"] + COLORS["bold"])}{colorize_text(TERMS_TEXT, COLORS["white"])}

{colorize_text("License:", COLORS["white"] + COLORS["bold"])}{colorize_text(LICENSE_TEXT, COLORS["white"])}

{colorize_text("Responsible Use:", COLORS["white"] + COLORS["bold"])}{colorize_text(RESPONSIBLE_USE_TEXT, COLORS["white"])}





{colorize_text("Press Enter to Continue...", COLORS["white"])}
"""
    print(cover_page)
    input() # Wait for Enter key press



def display_11ku7_logo_green():
    """Generates and displays only the 11KU7 AI Coder logo in green, with no other text."""

    # ASCII Art Logo (No changes needed)
    ascii_logo = """

                     ██╗ ██╗██╗  ██╗██╗   ██╗███████╗     █████╗ ██╗
                    ███║███║██║ ██╔╝██║   ██║╚════██║    ██╔══██╗██║
                    ╚██║╚██║█████╔╝ ██║   ██║    ██╔╝    ███████║██║
                     ██║ ██║██╔═██╗ ██║   ██║   ██╔╝     ██╔══██║██║
                     ██║ ██║██║  ██╗╚██████╔╝   ██║      ██║  ██║██║
                     ╚═╝ ╚═╝╚═╝  ╚═╝ ╚═════╝    ╚═╝      ╚═╝  ╚═╝╚═╝

                        ██████╗ ██████╗ ██████╗ ███████╗██████╗
                       ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗
                       ██║     ██║   ██║██║  ██║█████╗  ██████╔╝
                       ██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗
                       ╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║
                        ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝

    """

    # ANSI escape codes for colors (Simplified - Green and Reset)
    COLORS = {
        "green": "\033[32m",
        "reset": "\033[0m",
    }

    def clear_console_internal():
        os.system('cls' if os.name == 'nt' else 'clear')

    def get_color_choices_internal():
        """Hardcoded color choice - always green."""
        return ["green", "green"]

    def display_logo_hue_internal(color_names):
        """Displays the ASCII logo in green."""
        green_code = COLORS.get(color_names[0], COLORS["green"])

        colored_logo = ""
        lines = ascii_logo.splitlines()
        for line in lines:
            colored_logo += green_code + line + COLORS["reset"] + "\n"
        return colored_logo

    clear_console_internal()
    chosen_colors = get_color_choices_internal()
    logo_colored = display_logo_hue_internal(chosen_colors)

    # Display only the logo - no other text
    print(logo_colored)