"""Evaluation report helpers."""

from __future__ import annotations

import base64
from pathlib import Path

from .models import StrategyMetrics
from .pipeline import resolve_dataset_path


def metric_table(metrics: list[StrategyMetrics]) -> str:
    lines = ["| Strategy | Exact Row Accuracy | Notes |", "|---|---:|---|"]
    for metric in metrics:
        lines.append(f"| {metric.name} | {metric.exact_match_accuracy:.2%} | {metric.notes or '-'} |")
    return "\n".join(lines)


def field_accuracy_table(metric: StrategyMetrics) -> str:
    lines = ["| Field | Accuracy |", "|---|---:|"]
    for field, accuracy in metric.field_accuracies.items():
        lines.append(f"| {field} | {accuracy:.2%} |")
    return "\n".join(lines)


def write_markdown_report(
    report_path: Path,
    metrics: list[StrategyMetrics],
    selected_strategy: str,
    ops: dict[str, object],
) -> None:
    best = max(metrics, key=lambda item: item.exact_match_accuracy)
    content = "\n".join(
        [
            "# Evaluation Report",
            "",
            "## Strategy Comparison",
            "",
            metric_table(metrics),
            "",
            f"Selected final strategy: `{selected_strategy}`",
            "",
            "## Best Strategy Field Accuracy",
            "",
            field_accuracy_table(best),
            "",
            "## Operational Analysis",
            "",
            f"- Approximate logical model calls for the selected strategy: `{ops['approx_model_calls']}`",
            f"- Approximate input token usage for the selected strategy: `{ops['approx_input_tokens']}`",
            f"- Approximate output token usage for the selected strategy: `{ops['approx_output_tokens']}`",
            f"- Actual uncached provider request attempts during the evaluation run: `{ops['actual_uncached_provider_requests']}`",
            f"- Cache hits during the evaluation run: `{ops['cache_hits']}`",
            f"- Number of images processed by the selected strategy during this run: `{ops['images_processed']}`",
            f"- Cost assumption: {ops['cost_assumption']}",
            f"- Latency/runtime note: {ops['latency_note']}",
            f"- TPM/RPM note: {ops['rate_limit_note']}",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content + "\n", encoding="utf-8")


def _inline_image(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    suffix = path.suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if suffix == "jpg" else suffix
    return f"data:image/{mime};base64,{data}"


def write_html_report(
    report_path: Path,
    repo_root: Path,
    expected_rows: list[dict[str, str]],
    predicted_rows: list[dict[str, str]],
) -> None:
    cards: list[str] = []
    for index, (expected, predicted) in enumerate(zip(expected_rows, predicted_rows), start=1):
        image_path = expected["image_paths"].split(";")[0].strip()
        resolved = resolve_dataset_path(repo_root, image_path)
        image_tag = ""
        if resolved.exists():
            image_tag = f'<img src="{_inline_image(resolved)}" alt="sample {index}" loading="lazy" />'
        cards.append(
            f"""
            <article class="card" data-status="{predicted['claim_status']}">
              <header>
                <span class="index">Case {index:02d}</span>
                <span class="pill">{predicted['claim_status']}</span>
              </header>
              {image_tag}
              <p class="claim">{expected['user_claim']}</p>
              <div class="grid">
                <div><strong>Expected</strong><br>{expected['claim_status']} / {expected['issue_type']} / {expected['object_part']}</div>
                <div><strong>Predicted</strong><br>{predicted['claim_status']} / {predicted['issue_type']} / {predicted['object_part']}</div>
              </div>
              <p class="justification">{predicted['claim_status_justification']}</p>
            </article>
            """
        )
    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Claim Review Explorer</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --ink: #1d1b18;
      --card: #fffaf1;
      --accent: #9b3d2d;
      --muted: #756b5d;
      --line: #d9c9ad;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, 'Times New Roman', serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(155,61,45,.14), transparent 28%),
        linear-gradient(180deg, #faf6ed, var(--bg));
      min-height: 100vh;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 40px 20px 64px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2.2rem, 4vw, 4.5rem);
      line-height: .95;
      letter-spacing: -.04em;
    }}
    .sub {{
      margin: 0 0 28px;
      max-width: 62ch;
      color: var(--muted);
      font-size: 1.05rem;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: rgba(255,250,241,.92);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 16px;
      box-shadow: 0 14px 28px rgba(61,42,25,.08);
      backdrop-filter: blur(8px);
    }}
    .card header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      gap: 12px;
    }}
    .index {{
      font-size: .78rem;
      text-transform: uppercase;
      letter-spacing: .14em;
      color: var(--muted);
    }}
    .pill {{
      border: 1px solid rgba(155,61,45,.2);
      color: var(--accent);
      background: rgba(155,61,45,.08);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: .78rem;
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    img {{
      width: 100%;
      border-radius: 16px;
      display: block;
      margin-bottom: 12px;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      border: 1px solid rgba(0,0,0,.06);
    }}
    .claim {{
      font-size: .95rem;
      line-height: 1.55;
      color: #2d2a26;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin: 14px 0;
      font-size: .9rem;
    }}
    .justification {{
      margin: 0;
      padding-top: 12px;
      border-top: 1px solid rgba(0,0,0,.08);
      color: #433a30;
      line-height: 1.45;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Claim Review Explorer</h1>
    <p class="sub">Sample-set prediction drilldown for interview prep and README screenshots. Each card compares expected labels against the current pipeline output.</p>
    <section class="cards">
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
