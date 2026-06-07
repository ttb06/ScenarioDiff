#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_NAME = "gemini-2.5-flash"
DOMAIN_PROFILES = {
    "Economy": "international trade activity, measured in millions of U.S. dollars",
    "Energy": "gasoline prices, measured in U.S. dollars per gallon",
    "Security": "disaster and emergency assistance activity",
    "SocialGood": "the unemployment rate, measured as a percentage",
    "Traffic": "aggregate vehicle travel volume",
}
MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
TEXT_TIME_PATTERNS = [
    re.compile(r"\b(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b", re.I),
    re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.](?:19|20)?\d{2}\b", re.I),
    re.compile(rf"\b(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+(?:19|20)\d{{2}})?\b", re.I),
    re.compile(rf"\b(?:{MONTHS})\s+(?:19|20)\d{{2}}\b", re.I),
    re.compile(r"\b(?:19|20)\d{2}\b"),
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),
]
PROMPT_TIME_PATTERNS = TEXT_TIME_PATTERNS[:4] + TEXT_TIME_PATTERNS[5:]


def clean_text(value, max_chars=1200):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).replace("\n", " ").strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    for pattern in TEXT_TIME_PATTERNS:
        text = pattern.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;:-")
    return text[:max_chars]


def assert_timestamp_free(prompt):
    for pattern in PROMPT_TIME_PATTERNS:
        match = pattern.search(prompt)
        if match:
            raise ValueError(f"Prompt contains a calendar timestamp: {match.group(0)!r}")


def parse_json_object(text):
    text = (text or "").strip().replace("```json", "").replace("```", "")
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def format_value(value):
    return f"{float(value):.8g}"


def format_context(items, max_items=30, max_chars=1200):
    lines = []
    for index, item in enumerate(items[:max_items], start=1):
        fact = clean_text(item.get("fact"), max_chars)
        prediction = clean_text(item.get("preds"), max_chars)
        if not fact and not prediction:
            continue
        line = f"{index}. {fact}" if fact else f"{index}. {prediction}"
        if fact and prediction:
            line += f" Expected implication: {prediction}"
        lines.append(line)
    return "\n".join(lines) if lines else "[NO_EVENT]"


def historical_context_prompt(domain, context_block):
    prompt = f"""You are the Historical Context Agent for the {domain} domain.

Summarize only evidence relevant to the target series for one observed step.
Inputs intentionally omit calendar dates. Do not infer or add dates.

CONTEXT
{context_block}

Return exactly one JSON object:
{{"intrinsic":"concise expected or endogenous development","trigger":"concise unexpected or exogenous event"}}

Use "[NO_EVENT]" for intrinsic and an empty trigger when no useful evidence exists."""
    assert_timestamp_free(prompt)
    return prompt


def scenario_prompt(domain, profile, history_values, stepwise_context, horizon):
    history_lines = []
    context_lines = []
    length = len(history_values)
    for index, value in enumerate(history_values):
        relative_index = index - length + 1
        history_lines.append(f"{relative_index}: {format_value(value)}")
        summary = clean_text(stepwise_context[index], 500)
        context_lines.append(f"{relative_index}: {summary or '[NO_EVENT]'}")

    prompt = f"""You are the Scenario Agent for {domain}: {profile}.

Create a qualitative scenario for the next {horizon} steps. Use only the
relative sequence and observed evidence below. Inputs intentionally omit
calendar dates. Do not infer or add dates. Do not output numerical forecasts.

OBSERVED VALUES (relative index: value; index 0 is most recent)
{chr(10).join(history_lines)}

HISTORICAL CONTEXT (relative index: summary)
{chr(10).join(context_lines)}

Return one concise sentence describing direction, volatility, and the strongest
drivers expected over the forecast horizon."""
    assert_timestamp_free(prompt)
    return prompt


def anchor_prompt(domain, profile, history_values, stepwise_context, scenario, horizon):
    values = np.asarray(history_values, dtype=float)
    length = len(values)
    recent_lines = [
        f"{index - length + 1}: {format_value(value)}"
        for index, value in enumerate(values)
    ]
    context_lines = [
        f"{index - length + 1}: {clean_text(summary, 500) or '[NO_EVENT]'}"
        for index, summary in enumerate(stepwise_context)
    ]
    stats = (
        f"mean={format_value(values.mean())}, std={format_value(values.std() + 1e-6)}, "
        f"min={format_value(values.min())}, max={format_value(values.max())}"
    )
    scenario = clean_text(scenario, 800)

    prompt = f"""You are the Anchor Guidance Agent for {domain}: {profile}.

Propose sparse future anchor intervals only when observed evidence supports an
abrupt or event-driven change. Inputs intentionally omit calendar dates. Do not
infer or add dates.

OBSERVED VALUES (relative index: value; index 0 is most recent)
{chr(10).join(recent_lines)}

OBSERVED STATISTICS
{stats}

HISTORICAL CONTEXT (relative index: summary)
{chr(10).join(context_lines)}

SCENARIO
{scenario or '[NO_EVENT]'}

Return exactly one JSON object with at most five anchors:
{{"points":[{{"t":1,"f":0,"type":"NULL","v_lo":0.0,"v_hi":0.0,"confidence":0.5}}]}}

Rules:
- t is a relative future step in [1, {horizon}].
- Return {{"points":[]}} when evidence does not support a local constraint.
- Intervals use the original value scale and must satisfy v_lo <= v_hi.
- Confidence must be in [0.1, 1.0].
- Do not include explanations or extra keys."""
    assert_timestamp_free(prompt)
    return prompt


class GeminiClient:
    def __init__(self, api_keys, retries=4, sleep_seconds=2.0):
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError("Install google-generativeai to run the three-agent pipeline.") from exc
        self.genai = genai
        self.api_keys = [key.strip() for key in api_keys.split(",") if key.strip()]
        if not self.api_keys:
            raise ValueError("Set GEMINI_API_KEY or pass --api_key.")
        self.model_name = MODEL_NAME
        self.retries = retries
        self.sleep_seconds = sleep_seconds
        self.key_index = 0

    def generate(self, prompt, temperature, max_output_tokens, json_output=False):
        assert_timestamp_free(prompt)
        last_error = None
        for attempt in range(self.retries):
            try:
                key = self.api_keys[self.key_index % len(self.api_keys)]
                self.key_index += 1
                self.genai.configure(api_key=key)
                model = self.genai.GenerativeModel(self.model_name)
                config = {
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                }
                if json_output:
                    config["response_mime_type"] = "application/json"
                response = model.generate_content(prompt, generation_config=config)
                return (response.text or "").strip()
            except Exception as exc:
                last_error = exc
                time.sleep(self.sleep_seconds * (attempt + 1))
        raise RuntimeError(f"Gemini request failed after {self.retries} attempts") from last_error


def load_textual_csv(root, domain, kind):
    path = root / "textual" / domain / f"{domain}_{kind}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    for column in ("start_date", "end_date"):
        frame[column] = pd.to_datetime(frame[column])
    if "preds" not in frame:
        frame["preds"] = ""
    return frame


def context_slice(report, search, start, end, max_items):
    frames = []
    for frame in (report, search):
        selected = frame.loc[(frame["end_date"] >= start) & (frame["end_date"] <= end)].copy()
        frames.append(selected)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("end_date", ascending=False, kind="mergesort")
    return combined.head(max_items).to_dict("records")


def load_csv_map(path, value_columns):
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    return {
        str(row["end_date"]): tuple(clean_text(row.get(column), 4000) for column in value_columns)
        for _, row in frame.iterrows()
    }


def write_csv_map(path, mapping, value_columns):
    rows = []
    for key in sorted(mapping):
        values = mapping[key]
        rows.append({"end_date": key, **dict(zip(value_columns, values))})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["end_date", *value_columns]).to_csv(path, index=False)


def load_jsonl_map(path):
    mapping = {}
    if not path.exists():
        return mapping
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("end_date", ""))
            if key:
                mapping[key] = row
    return mapping


def write_jsonl_map(path, mapping):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(mapping):
            handle.write(json.dumps(mapping[key], ensure_ascii=True) + "\n")
    os.replace(temporary, path)


def stepwise_summaries(history, historical_map):
    summaries = []
    for end_date in history["end_date"]:
        intrinsic, trigger = historical_map.get(str(end_date.date()), ("[NO_EVENT]", ""))
        summary = intrinsic
        if trigger:
            summary = f"{summary} Trigger: {trigger}"
        summaries.append(summary)
    return summaries


def split_window_indices(length, lookback, horizon, split):
    train_end = int(length * 0.7)
    validation_end = length - int(length * 0.2)
    bounds = {
        "all": (lookback, length - horizon),
        "train": (lookback, train_end - horizon),
        "val": (train_end, validation_end - horizon),
        "test": (validation_end, length - horizon),
        "valtest": (train_end, length - horizon),
    }
    first, last = bounds[split]
    return list(range(first, last + 1)) if last >= first else []


def normalize_points(payload, horizon):
    points = payload.get("points", []) if isinstance(payload, dict) else []
    normalized = []
    for point in points:
        if not isinstance(point, dict):
            continue
        try:
            step = int(point["t"])
            lower = float(point["v_lo"])
            upper = float(point["v_hi"])
            confidence = float(point.get("confidence", 0.5))
        except (KeyError, TypeError, ValueError):
            continue
        if not 1 <= step <= horizon:
            continue
        lower, upper = sorted((lower, upper))
        normalized.append({
            "t": step,
            "f": 0,
            "type": "NULL",
            "v_lo": lower,
            "v_hi": upper,
            "confidence": min(1.0, max(0.1, confidence)),
        })
    return normalized[:5]


def run_phase_1(args, client, root, domain, numerical, report, search):
    output = root / "textual" / domain / f"{domain}_intrinsic_trigger.csv"
    cache = load_csv_map(output, ("intrinsic", "trigger")) if args.resume and output.exists() else {}
    for count, (_, row) in enumerate(numerical.iterrows(), start=1):
        key = str(row["end_date"].date())
        if key in cache:
            continue
        items = context_slice(report, search, row["start_date"], row["end_date"], args.max_context_items)
        prompt = historical_context_prompt(domain, format_context(items, args.max_context_items, args.max_context_chars))
        response = client.generate(prompt, args.temperature, args.max_output_tokens, json_output=True)
        payload = parse_json_object(response)
        intrinsic = clean_text(payload.get("intrinsic"), 1000) or "[NO_EVENT]"
        trigger = clean_text(payload.get("trigger"), 1000)
        cache[key] = (intrinsic, trigger)
        if count % args.flush_every == 0:
            write_csv_map(output, cache, ("intrinsic", "trigger"))
    write_csv_map(output, cache, ("intrinsic", "trigger"))
    return cache


def run_phase_2(args, client, root, domain, profile, numerical, historical_map):
    output = root / "textual" / domain / f"{domain}_coarse_pred.csv"
    cache = load_csv_map(output, ("coarse_pred",)) if args.resume and output.exists() else {}
    for count, origin in enumerate(range(args.seq_len, len(numerical) + 1), start=1):
        history = numerical.iloc[origin - args.seq_len:origin]
        key = str(history.iloc[-1]["end_date"].date())
        if key in cache:
            continue
        summaries = stepwise_summaries(history, historical_map)
        prompt = scenario_prompt(domain, profile, history[args.target].tolist(), summaries, args.pred_len)
        response = client.generate(prompt, args.temperature, args.max_output_tokens)
        cache[key] = (clean_text(response, 2000) or "[NO_EVENT]",)
        if count % args.flush_every == 0:
            write_csv_map(output, cache, ("coarse_pred",))
    write_csv_map(output, cache, ("coarse_pred",))
    return cache


def run_phase_3(args, client, root, domain, profile, numerical, historical_map, scenario_map):
    suffix = "" if args.split == "all" else f"_{args.split}"
    output = root / "textual" / domain / f"{domain}_abnormal_points{suffix}.jsonl"
    cache = load_jsonl_map(output) if args.resume else {}
    origins = split_window_indices(len(numerical), args.seq_len, args.pred_len, args.split)
    if args.max_windows:
        origins = origins[:args.max_windows]
    for count, origin in enumerate(origins, start=1):
        history = numerical.iloc[origin - args.seq_len:origin]
        key = str(history.iloc[-1]["end_date"].date())
        if key in cache:
            continue
        summaries = stepwise_summaries(history, historical_map)
        scenario = scenario_map.get(key, ("[NO_EVENT]",))[0]
        prompt = anchor_prompt(
            domain,
            profile,
            history[args.target].tolist(),
            summaries,
            scenario,
            args.pred_len,
        )
        response = client.generate(prompt, args.temperature, args.max_output_tokens, json_output=True)
        cache[key] = {"end_date": key, "points": normalize_points(parse_json_object(response), args.pred_len)}
        if count % args.flush_every == 0:
            write_jsonl_map(output, cache)
    write_jsonl_map(output, cache)


def build_parser():
    parser = argparse.ArgumentParser(description="Generate the three cached ScenarioDiff guidance levels.")
    parser.add_argument("--root_path", default="Time-MMD")
    parser.add_argument("--data_path", required=True, help="For example: Economy/Economy.csv")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--phase", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--split", choices=("all", "train", "val", "test", "valtest"), default="all")
    parser.add_argument("--api_key", default=os.environ.get("GEMINI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_output_tokens", type=int, default=2048)
    parser.add_argument("--max_context_items", type=int, default=30)
    parser.add_argument("--max_context_chars", type=int, default=1200)
    parser.add_argument("--max_windows", type=int, default=0)
    parser.add_argument("--flush_every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    root = Path(args.root_path).expanduser().resolve()
    data_path = Path(args.data_path)
    domain = data_path.parts[0]
    if domain not in DOMAIN_PROFILES:
        raise ValueError(f"Unsupported domain {domain!r}; expected one of {sorted(DOMAIN_PROFILES)}")

    numerical = pd.read_csv(root / "numerical" / data_path)
    for column in ("start_date", "end_date"):
        numerical[column] = pd.to_datetime(numerical[column])
    numerical = numerical.dropna(subset=[args.target]).sort_values("end_date", kind="mergesort").reset_index(drop=True)
    report = load_textual_csv(root, domain, "report")
    search = load_textual_csv(root, domain, "search")
    client = GeminiClient(args.api_key)
    profile = DOMAIN_PROFILES[domain]

    run_all = args.phase == 0
    historical_path = root / "textual" / domain / f"{domain}_intrinsic_trigger.csv"
    scenario_path = root / "textual" / domain / f"{domain}_coarse_pred.csv"

    historical_map = (
        run_phase_1(args, client, root, domain, numerical, report, search)
        if run_all or args.phase == 1
        else load_csv_map(historical_path, ("intrinsic", "trigger"))
    )
    if run_all or args.phase == 2:
        scenario_map = run_phase_2(args, client, root, domain, profile, numerical, historical_map)
    elif args.phase == 3:
        scenario_map = load_csv_map(scenario_path, ("coarse_pred",))
    else:
        scenario_map = {}
    if run_all or args.phase == 3:
        run_phase_3(args, client, root, domain, profile, numerical, historical_map, scenario_map)


if __name__ == "__main__":
    main()
