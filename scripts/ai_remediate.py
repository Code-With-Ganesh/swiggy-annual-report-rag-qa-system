import argparse
import json
from pathlib import Path

AI_PROMPT = """You are a cloud security engineer. Analyze this Trivy Terraform scan JSON and produce:
1) A summary of high/critical risks.
2) A plain-English explanation of impact.
3) Concrete Terraform code changes to remediate.
4) A secure final version aligned to least privilege and encryption-by-default.
"""


def load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Report file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_findings(report: dict) -> list[dict]:
    findings: list[dict] = []
    for result in report.get("Results", []):
        for misconfig in result.get("Misconfigurations", []):
            severity = misconfig.get("Severity", "UNKNOWN")
            if severity in {"HIGH", "CRITICAL"}:
                findings.append(
                    {
                        "id": misconfig.get("ID", "N/A"),
                        "title": misconfig.get("Title", "No title"),
                        "severity": severity,
                        "description": misconfig.get("Description", "No description"),
                        "resolution": misconfig.get("Resolution", "No remediation provided"),
                        "file": result.get("Target", "unknown"),
                    }
                )
    return findings


def print_console_report(findings: list[dict], insecure_file: Path, secure_file: Path) -> None:
    print("=" * 72)
    print("AI SECURITY ANALYSIS LOG")
    print("=" * 72)
    print("PROMPT USED:")
    print(AI_PROMPT.strip())
    print("-" * 72)

    if not findings:
        print("No HIGH/CRITICAL findings were detected in the provided report.")
        return

    print(f"Identified {len(findings)} HIGH/CRITICAL findings:\n")
    for idx, finding in enumerate(findings, start=1):
        print(f"{idx}. [{finding['severity']}] {finding['id']} - {finding['title']}")
        print(f"   File: {finding['file']}")
        print(f"   Risk: {finding['description']}")
        print(f"   Remediation: {finding['resolution']}")
        print()

    print("Recommended code remediation summary:")
    print("- Restrict SSH ingress from 0.0.0.0/0 to a trusted admin CIDR.")
    print("- Remove publicly exposed management ports unless strictly required.")
    print("- Enforce encrypted root volumes for EC2 instances.")
    print("- Keep least-privilege inbound rules and auditable variable validation.")
    print()
    print(f"Insecure reference code: {insecure_file}")
    print(f"Secured Terraform code: {secure_file}")
    print("Apply the secure stack and rerun Trivy to confirm zero HIGH/CRITICAL issues.")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-style analysis for Trivy Terraform report")
    parser.add_argument("--report", required=True, help="Path to Trivy JSON report")
    parser.add_argument("--insecure-file", required=True, help="Path to insecure Terraform file")
    parser.add_argument("--secure-file", required=True, help="Path to secure Terraform file")
    args = parser.parse_args()

    report = load_report(Path(args.report))
    findings = collect_findings(report)
    print_console_report(findings, Path(args.insecure_file), Path(args.secure_file))


if __name__ == "__main__":
    main()
