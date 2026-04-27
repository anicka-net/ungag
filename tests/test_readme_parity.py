"""Check README model tables stay in sync with KNOWN_RECIPES and shipped directions."""

from __future__ import annotations

import re
from pathlib import Path

from ungag.recipes import KNOWN_RECIPES

README = Path(__file__).resolve().parent.parent / "README.md"


def _readme_text() -> str:
    return README.read_text()


def _parse_table_rows(text: str, header_pattern: str) -> list[str]:
    """Extract non-header rows from the first markdown table after *header_pattern*."""
    start = text.find(header_pattern)
    if start == -1:
        return []
    lines = text[start:].splitlines()
    rows: list[str] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if "---" in stripped:
                in_table = True
                continue
            if in_table:
                rows.append(stripped)
        elif in_table:
            break
    return rows


def _model_names_from_table(header: str) -> set[str]:
    """Return model names (first column) from a README table."""
    rows = _parse_table_rows(_readme_text(), header)
    names: set[str] = set()
    for row in rows:
        cells = [c.strip() for c in row.split("|")]
        if len(cells) >= 2:
            name = cells[1].strip()
            if name:
                names.add(name)
    return names


def _condition_dependent_from_recipes() -> set[str]:
    """Return names of models with projection_result == 'condition_dependent'."""
    return {
        r["name"] for r in KNOWN_RECIPES.values()
        if r.get("projection_result") == "condition_dependent"
    }


def test_condition_dependent_table_matches_recipes():
    """Every condition_dependent recipe must appear in the README table."""
    readme_names = _model_names_from_table("**Condition-dependent under projection**")
    recipe_names = _condition_dependent_from_recipes()

    missing = recipe_names - readme_names
    assert not missing, (
        f"condition_dependent recipes missing from README: {sorted(missing)}"
    )


def test_shipped_directions_table():
    """Every model in the shipped directions table should have a direction file."""
    import ungag
    table_names = _model_names_from_table("Pre-extracted directions bundled")
    direction_keys = set(ungag.DIRECTIONS.keys())

    for row in _parse_table_rows(_readme_text(), "Pre-extracted directions bundled"):
        cells = [c.strip() for c in row.split("|")]
        if len(cells) >= 2:
            key = cells[1].strip().strip("`")
            if key:
                assert key in direction_keys, (
                    f"README lists direction key `{key}` but it's not in ungag.DIRECTIONS"
                )


def test_model_count_consistent():
    """The '4 models out of N tested' count in the intro should be consistent."""
    text = _readme_text()
    match = re.search(r"(\d+)\s+models?\s+out\s+of\s+(\d+)\s+tested", text)
    assert match, "Could not find 'N models out of M tested' in README"
    success_count = int(match.group(1))
    total_count = int(match.group(2))

    condition_dependent = _condition_dependent_from_recipes()
    assert success_count == len(condition_dependent), (
        f"README says {success_count} successful but recipes has "
        f"{len(condition_dependent)} condition_dependent"
    )
    assert total_count >= success_count, (
        f"README says {total_count} tested but only {success_count} successful"
    )
