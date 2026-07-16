"""
Artifact quality audit for the trait vector artifact bank.

Loads configs/trait_vector_artifacts.yaml, runs all quality checks,
prints a human-readable report, and saves:
    data/processed/vector_artifacts/vector_artifact_audit.csv
    data/processed/vector_artifacts/vector_artifact_audit.md  (optional)

Usage:
    python scripts/audit_vector_artifacts.py
    python scripts/audit_vector_artifacts.py --no-md
    python scripts/audit_vector_artifacts.py --artifacts-path path/to/yaml
    python scripts/audit_vector_artifacts.py --out-dir path/to/dir
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.vectors.artifact_bank import (  # noqa: E402
    load_artifact_bank,
    load_artifact_bank_flexible,
)
from src.vectors.artifact_quality import (  # noqa: E402
    AuditFinding,
    findings_to_df,
    run_quality_checks,
)

_SEVERITY_ORDER = {"high": 0, "warning": 1, "info": 2}
_SEVERITY_LABEL = {"high": "🔴 HIGH", "warning": "🟡 WARN", "info": "🔵 INFO"}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _sep(char: str = "─", width: int = 68) -> str:
    return char * width


def _print_banner() -> None:
    print("=" * 68)
    print("  Trait Vector Artifact Bank — Quality Audit")
    print("=" * 68)
    print()
    print("  Goal: catch confounds and weak design before vector extraction.")
    print("  These flags are for human review — no artifact is auto-rejected.")
    print()
    print("  Common failure modes this audit guards against:")
    print("    • Positive pole = generic helpful/good rather than specific trait")
    print("    • Negative pole introduces extra traits (cruelty in honesty,")
    print("      deception in harmlessness, harm in compassion, etc.)")
    print("    • Elicitation questions name the construct they probe")
    print("    • Extraction / validation questions overlap in content")
    print("    • Near-duplicate prompts reduce contrastive diversity")
    print("    • Vector captures generic moral valence, not the target trait")
    print()


def _severity_sort_key(f: AuditFinding) -> tuple:
    return (_SEVERITY_ORDER.get(f.severity, 99), f.trait, f.artifact_type)


def _print_summary(findings: list[AuditFinding]) -> None:
    counts = Counter(f.severity for f in findings)
    by_trait = Counter(f.trait for f in findings)
    by_type = Counter(f.issue_type for f in findings)

    print(_sep("="))
    print("  Summary")
    print(_sep("─"))
    total = len(findings)
    print(f"  Total findings  : {total}")
    for sev in ("high", "warning", "info"):
        n = counts.get(sev, 0)
        label = _SEVERITY_LABEL[sev]
        bar = "█" * min(n, 30)
        print(f"  {label:<14}  {n:>3}  {bar}")

    if total > 0:
        print()
        print("  Findings by trait:")
        for trait in sorted(by_trait):
            print(f"    {trait:<16} {by_trait[trait]:>3}")
        print()
        print("  Findings by issue type:")
        for issue, n in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"    {issue:<40} {n:>3}")
    print()


def _print_findings(findings: list[AuditFinding]) -> None:
    if not findings:
        print("  ✓ No issues found.")
        return

    sorted_findings = sorted(findings, key=_severity_sort_key)

    current_severity = None
    for f in sorted_findings:
        if f.severity != current_severity:
            current_severity = f.severity
            label = _SEVERITY_LABEL[f.severity]
            print(_sep("─"))
            print(f"  {label} findings")
            print(_sep("─"))

        print(f"\n  [{f.trait}] {f.artifact_type} / {f.artifact_id}")
        print(f"  Issue  : {f.issue_type}")
        print(f"  Text   : {f.text}")
        print(f"  Why    : {f.explanation}")
        print(f"  Action : {f.suggested_review_action}")


def _write_md_report(
    findings: list[AuditFinding], out_path: Path, artifacts_path: str
) -> None:
    counts = Counter(f.severity for f in findings)
    lines: list[str] = [
        "# Trait Vector Artifact Bank — Quality Audit Report",
        "",
        f"> Source: `{artifacts_path}`  ",
        f"> Total findings: {len(findings)} "
        f"({counts.get('high', 0)} high / "
        f"{counts.get('warning', 0)} warning / "
        f"{counts.get('info', 0)} info)",
        "",
        "## Common failure modes audited",
        "",
        "| Failure mode | Risk |",
        "|---|---|",
        "| Positive pole = generic helpfulness | Vector captures 'assistant goodness' not specific trait |",
        "| Negative pole introduces other traits | Vector partially captures a different moral dimension |",
        "| Questions name the construct | Model is cued about what is being evaluated |",
        "| Extraction/validation text overlap | Validation is not genuinely held out |",
        "| Near-duplicate prompts | Reduced contrastive diversity |",
        "| Generic moral valence dominance | Vector measures good-vs-evil, not the target trait |",
        "",
        "---",
        "",
    ]

    if not findings:
        lines += ["## Findings", "", "No issues found.", ""]
    else:
        sorted_findings = sorted(findings, key=_severity_sort_key)
        current_severity = None
        for f in sorted_findings:
            if f.severity != current_severity:
                current_severity = f.severity
                label = {"high": "🔴 High", "warning": "🟡 Warning", "info": "🔵 Info"}[f.severity]
                lines += ["", f"## {label} severity", ""]

            lines += [
                f"### `[{f.trait}]` {f.artifact_id}",
                "",
                f"- **Artifact type**: {f.artifact_type}",
                f"- **Issue**: {f.issue_type}",
                f"- **Text**: {f.text}",
                f"- **Why**: {f.explanation}",
                f"- **Action**: {f.suggested_review_action}",
                "",
            ]

    lines += [
        "---",
        "",
        "*These findings are for human review. No artifact is automatically rejected.*",
        "*Address high-severity findings before proceeding to Stage 2B.*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit trait vector artifacts for quality and confound risks."
    )
    parser.add_argument(
        "--artifacts-path",
        default=str(_ROOT / "configs" / "trait_vector_artifacts.yaml"),
        help="Path to trait_vector_artifacts.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_ROOT / "data" / "processed" / "vector_artifacts"),
        help="Output directory for audit files",
    )
    parser.add_argument(
        "--no-md",
        action="store_true",
        help="Skip Markdown report output",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _print_banner()

    try:
        bank = load_artifact_bank(args.artifacts_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"  Loaded: {args.artifacts_path}\n")

    findings = run_quality_checks(bank)
    findings_sorted = sorted(findings, key=_severity_sort_key)

    _print_summary(findings_sorted)
    _print_findings(findings_sorted)

    # --- Save CSV ---
    csv_path = out_dir / "vector_artifact_audit.csv"
    df = findings_to_df(findings_sorted)
    df.to_csv(csv_path, index=False)

    print()
    print(_sep("="))
    print("  Saved audit files:")
    print(f"    CSV    : {csv_path}")

    if not args.no_md:
        md_path = out_dir / "vector_artifact_audit.md"
        _write_md_report(findings_sorted, md_path, args.artifacts_path)
        print(f"    MD     : {md_path}")

    print()

    # Exit with non-zero if any high-severity findings
    high_count = sum(1 for f in findings if f.severity == "high")
    if high_count > 0:
        print(f"  ⚠  {high_count} HIGH severity finding(s) — review before Stage 2B.")
    else:
        print("  ✓ No HIGH severity findings.")

    print(_sep("="))

    if high_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
