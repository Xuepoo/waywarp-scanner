"""Main executable script wrapper for waywarp-scanner CLI."""

from waywarp_scanner.cli import cli


def main() -> None:
    """Execute standard Click command group entry point."""
    cli()


if __name__ == "__main__":
    main()
