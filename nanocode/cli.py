import os
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from .agent import Agent, AgentObserver

class CommandCompleter(Completer):
    def __init__(self, commands, models):
        self.commands = commands
        self.models = models

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/model "):
            prefix = text[7:]
            for m in self.models:
                if m.startswith(prefix):
                    yield Completion(m, start_position=-len(prefix))
        elif text.startswith("/"):
            for cmd in self.commands:
                for name in cmd["names"]:
                    if name.startswith(text):
                        yield Completion(name, start_position=-len(text))

class CLI(AgentObserver):
    def __init__(self, agent: Agent):
        self.agent = agent
        self.console = Console()
        self.session = PromptSession()
        self.spinner_active = False
        self.printed_prefix = False
        self.live_spinner = None
        self.commands = [
            {"names": ["/q", "/quit", "/exit"], "desc": "Quit application", "handler": self.handle_quit},
            {"names": ["/c"], "desc": "Clear current conversation context", "handler": self.handle_clear},
            {"names": ["/models"], "desc": "List available models", "handler": self.handle_models},
            {"names": ["/model"], "desc": "Switch LLM model", "handler": self.handle_model, "args": True},
            {"names": ["/save", "/s"], "desc": "Save conversation to JSON", "handler": self.handle_save, "args": True},
            {"names": ["/load", "/l"], "desc": "Load conversation from JSON", "handler": self.handle_load, "args": True},
        ]
        self.known_models = [
            "gemini-2.0-flash", "gemini-2.0-pro-exp-02-05", 
            "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
            "gpt-4o", "gpt-4o-mini"
        ]

    def update_status_line(self):
        status = Text.assemble(
            (f"nanocode-py", "bold"),
            " | ",
            (f"{self.agent.current_model.id}", "cyan"),
            " | ",
            (f"{os.getcwd()}", "dim")
        )
        self.console.print(status)

    def print_help(self):
        self.console.print("[dim]Available commands:[/dim]")
        for cmd in self.commands:
            names = ", ".join(cmd["names"])
            args = " <args>" if cmd.get("args") else ""
            self.console.print(f"  [dim]{names}{args} - {cmd['desc']}[/dim]")

    async def run(self):
        self.update_status_line()
        self.print_help()
        self.console.print()

        completer = CommandCompleter(self.commands, self.known_models)

        while True:
            self.console.print("[dim]────────────────────────────────────────────────────────────────────────────────[/dim]")
            try:
                user_input = await self.session.prompt_async(HTML("<b><blue>❯ </blue></b>"), completer=completer)
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input.strip():
                continue

            self.console.print("[dim]────────────────────────────────────────────────────────────────────────────────[/dim]")

            if user_input.startswith("/"):
                parts = user_input.split(" ", 1)
                cmd_name = parts[0]
                cmd_args = parts[1] if len(parts) > 1 else ""

                found = False
                for cmd in self.commands:
                    if cmd_name in cmd["names"]:
                        if not await cmd["handler"](cmd_args):
                            return
                        found = True
                        break
                if not found:
                    self.console.print(f"[red]⏺ Unknown command: {cmd_name}[/red]")
                continue

            self.printed_prefix = False
            await self.agent.process_message(user_input, self)
            self.console.print()

    # AgentObserver implementation

    def on_thought_start(self):
        if self.spinner_active: return
        self.spinner_active = True
        self.live_spinner = Live(Spinner("dots", text=Text("Thinking...", style="dim")), console=self.console, transient=True)
        self.live_spinner.start()

    def on_thought_end(self):
        if self.spinner_active:
            self.spinner_active = False
            if self.live_spinner:
                self.live_spinner.stop()
                self.live_spinner = None

    def on_text_chunk(self, chunk: str):
        self.on_thought_end()
        if not self.printed_prefix:
            self.console.print("\n[cyan]⏺[/cyan] ", end="")
            self.printed_prefix = True
        self.console.print(chunk, end="")

    def on_tool_start(self, name: str, args_preview: str):
        self.on_thought_end()
        self.console.print(f"\n[green]⏺ {name}[/green]([dim]{args_preview}[/dim])")

    def on_tool_result(self, result_preview: str):
        self.console.print(f"  [dim]⎿  {result_preview}[/dim]")

    def on_error(self, error_msg: str):
        self.on_thought_end()
        self.console.print(f"\n[red]⏺ Error: {error_msg}[/red]")

    # Handlers

    @staticmethod
    async def handle_quit(_args: str) -> bool:
        return False

    async def handle_clear(self, _args: str) -> bool:
        self.agent.clear_history()
        self.console.print("[green]⏺ Cleared conversation[/green]")
        return True

    async def handle_models(self, _args: str) -> bool:
        models = await self.agent.list_available_models()
        if not models:
            self.console.print("  (No models found or error)")
        else:
            last_provider = ""
            for m in models:
                provider, model_id = m.split(":", 1)
                if provider != last_provider:
                    self.console.print(f"[yellow]⏺ Models for {provider}:[/yellow]")
                    last_provider = provider
                self.console.print(f"  - {model_id}")
        return True

    async def handle_model(self, args: str) -> bool:
        if not args:
            self.console.print("[red]⏺ Error: /model requires a name[/red]")
            return True
        try:
            new_model = await self.agent.switch_model(args)
            self.update_status_line()
            self.console.print(f"[green]⏺ Switched model to: {new_model}[/green]")
        except Exception as e:
            self.console.print(f"[red]⏺ Error: {str(e)}[/red]")
        return True

    async def handle_save(self, args: str) -> bool:
        if not args:
            self.console.print("[red]⏺ Error: /save requires a filename[/red]")
            return True
        try:
            self.agent.save_session(args)
            self.console.print(f"[green]⏺ Saved conversation to {args}[/green]")
        except Exception as e:
            self.console.print(f"[red]⏺ Error: {str(e)}[/red]")
        return True

    async def handle_load(self, args: str) -> bool:
        if not args:
            self.console.print("[red]⏺ Error: /load requires a filename[/red]")
            return True
        try:
            self.agent.load_session(args)
            self.update_status_line()
            self.console.print(f"[green]⏺ Loaded conversation from {args}[/green]")
        except Exception as e:
            self.console.print(f"[red]⏺ Error: {str(e)}[/red]")
        return True
