import time
import argparse
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live

console = Console()

def generate_dashboard(hub_url: str) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="advisories", size=8)
    )

    try:
        with httpx.Client(timeout=2.0) as client:
            summary_resp = client.get(f"{hub_url.rstrip('/')}/api/v1/analytics/summary")
            summaries = summary_resp.json() if summary_resp.status_code == 200 else []

            events_resp = client.get(f"{hub_url.rstrip('/')}/api/v1/analytics/events?limit=8")
            events = events_resp.json() if events_resp.status_code == 200 else []
    except Exception:
        summaries, events = [], []

    # Header
    total_spend = sum(s.get("total_spend_usd", 0.0) for s in summaries)
    total_tokens = sum(s.get("total_input_tokens", 0) + s.get("total_output_tokens", 0) for s in summaries)
    header_text = f"[bold cyan]Token Watch Telemetry Hub[/bold cyan] | [green]ONLINE[/green] | Total Spend: [yellow]${total_spend:.4f}[/yellow] | Total Tokens: [magenta]{total_tokens:,}[/magenta]"
    layout["header"].update(Panel(header_text, style="blue"))

    # Main: Agent Breakdown Table
    table = Table(title="Agent Token Metrics", expand=True)
    table.add_column("Agent ID", style="cyan", no_wrap=True)
    table.add_column("Invocations", justify="right")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Cache Hit %", justify="right", style="green")
    table.add_column("Spend (USD)", justify="right", style="yellow")
    table.add_column("Avg Latency", justify="right")

    for s in summaries:
        cache_pct = f"{s.get('cache_hit_ratio', 0.0) * 100:.1f}%"
        table.add_row(
            s.get("agent_id", "-"),
            str(s.get("invocation_count", 0)),
            f"{s.get('total_input_tokens', 0):,}",
            f"{s.get('total_output_tokens', 0):,}",
            cache_pct,
            f"${s.get('total_spend_usd', 0.0):.4f}",
            f"{s.get('avg_latency_ms', 0.0):.0f} ms"
        )
    layout["main"].update(table)

    # Advisories Panel
    adv_lines = []
    if events:
        latest = events[0]
        tools_dec = len(latest.get("tools_declared", []))
        tools_inv = len(latest.get("tools_invoked", []))
        if tools_dec > 5 and tools_inv <= 1:
            adv_lines.append(f"[bold yellow]⚠️ TOOL PRUNING:[/bold yellow] Agent '{latest.get('agent_id')}' has {tools_dec - tools_inv} unused tools declared. Consider dynamic pruning.")
        if latest.get("cached_read_tokens", 0) == 0 and latest.get("input_tokens", 0) > 1000:
            adv_lines.append(f"[bold blue]ℹ️ CACHE ALIGNMENT:[/bold blue] 0 cached tokens on large prompt for '{latest.get('agent_id')}'. Stabilize prefix order.")

    if not adv_lines:
        adv_text = "[green]All agent prompts running within optimal token efficiency thresholds.[/green]"
    else:
        adv_text = "\n".join(adv_lines)

    layout["advisories"].update(Panel(adv_text, title="Active Advisories & Optimization Alerts", style="yellow"))
    return layout

def run_dashboard(hub_url: str):
    with Live(generate_dashboard(hub_url), refresh_per_second=1, screen=True) as live:
        try:
            while True:
                time.sleep(1)
                live.update(generate_dashboard(hub_url))
        except KeyboardInterrupt:
            console.print("[yellow]Exiting Token Watch Dashboard.[/yellow]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token Watch CLI")
    subparsers = parser.add_subparsers(dest="command")

    dash_parser = subparsers.add_parser("dashboard", help="Start real-time TUI dashboard")
    dash_parser.add_argument("--hub-url", type=str, default="http://localhost:9000")

    args = parser.parse_args()
    if args.command == "dashboard":
        run_dashboard(args.hub_url)
    else:
        parser.print_help()
