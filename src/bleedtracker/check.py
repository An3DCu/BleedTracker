"""Comparing rebuilt results with my original ones.

Only values that exist in the reference are compared. Every one has to match exactly
at the precision it was stored with. Numbers are counted for the summary. Text and
true or false values are compared too. A difference in them is listed.
"""

from dataclasses import dataclass, field


@dataclass
class Comparison:
    """How many numbers were compared, how many matched, and every difference."""

    numbers: int = 0
    matching: int = 0
    differences: list[str] = field(default_factory=list)

    @property
    def all_match(self) -> bool:
        return not self.differences

    def summary(self) -> str:
        return f"{self.matching} of {self.numbers} numbers match my original results."


NOT_RESULTS = {"about"}


def compare_with_reference(results: dict, reference: dict) -> Comparison:
    """Compare every result section. The reference's own description is skipped."""
    return compare(results, {k: v for k, v in reference.items() if k not in NOT_RESULTS})


def is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def compare(got, expected, path: str = "", result: Comparison | None = None) -> Comparison:
    """Walk the reference and compare each value with the rebuilt one."""
    result = result if result is not None else Comparison()
    if isinstance(expected, dict):
        if not isinstance(got, dict):
            result.differences.append(f"{path or 'results'}: expected a section, got {got!r}")
            return result
        for key, value in expected.items():
            where = f"{path}.{key}" if path else key
            if key not in got:
                result.differences.append(f"{where}: missing")
                continue
            compare(got[key], value, where, result)
        return result
    if isinstance(expected, list):
        if not isinstance(got, list) or len(got) != len(expected):
            result.differences.append(f"{path}: expected {len(expected)} items")
            return result
        for i, (g, e) in enumerate(zip(got, expected, strict=True)):
            compare(g, e, f"{path}[{i}]", result)
        return result
    if is_number(expected):
        result.numbers += 1
    if got == expected:
        if is_number(expected):
            result.matching += 1
    else:
        result.differences.append(f"{path}: original {expected!r}, rebuilt {got!r}")
    return result
