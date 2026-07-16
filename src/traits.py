"""Trait-space helpers: iteration, lookup, and display."""

from __future__ import annotations

from src.config import TraitDefinition, TraitSpace


def get_trait(space: TraitSpace, name: str) -> TraitDefinition:
    if name not in space.traits:
        raise KeyError(f"Unknown trait '{name}'. Available: {space.names}")
    return space.traits[name]


def trait_summary_lines(space: TraitSpace) -> list[str]:
    """One-line display summary per trait (for logging/check_setup)."""
    lines = []
    for name, td in space.traits.items():
        kw_count = len(td.example_keywords_and_patterns)
        lines.append(f"  {name:<14} | {kw_count} example keywords/patterns")
    return lines
