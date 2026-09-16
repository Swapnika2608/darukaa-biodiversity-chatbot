"""
main.py
CLI entry point for the Darukaa.Earth Biodiversity Intelligence Chatbot.
Run: python main.py
"""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from chatbot import DarukaaChatbot

console = Console()

BANNER = """
[bold green]╔══════════════════════════════════════════════════════╗
║        🌿  DARUKAA.EARTH  –  Biodiversity AI         ║
║          AI Environmental Scientist Chatbot           ║
╚══════════════════════════════════════════════════════╝[/bold green]

[dim]Commands:  'reset' – clear memory & data  |  'quit' – exit[/dim]
[dim]Tip: Paste a JSON block to load structured data, e.g.:[/dim]
[dim]{"soil_organic_carbon_pct": 0.3, "rainfall_mm_year": 280, "land_use": "monoculture", "temperature_c": 28}[/dim]
"""

def main():
    console.print(BANNER)
    bot = DarukaaChatbot()

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye.[/yellow]")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            console.print("[yellow]Goodbye.[/yellow]")
            break
        if user_input.lower() == "reset":
            bot.reset()
            console.print("[green]Session reset.[/green]")
            continue

        with console.status("[bold green]Darukaa is thinking...[/bold green]"):
            response = bot.chat(user_input)

        console.print(Panel(Markdown(response), title="[bold green]Darukaa[/bold green]", border_style="green"))


if __name__ == "__main__":
    main()
