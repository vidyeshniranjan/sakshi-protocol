# =============================================================================
# run_benchmark.py — Sakshi-Protocol V3 benchmark runner
#
# Usage:
#   python run_benchmark.py --mode baseline --model gpt-5.5
#   python run_benchmark.py --mode sakshi --model claude-sonnet-4-6
#   python run_benchmark.py --mode sakshi_omega --model llama-3.3-70b
#   python run_benchmark.py --mode sakshi_omega --model qwen-3.5-9b
#   python run_benchmark.py --mode sakshi_omega --model gpt-5.5 --prompts prompts_500.json
#
# Modes:
#   baseline      — raw generation, no Sakshi signals, no Omega
#   sakshi        — full Sakshi signals + controller, Omega disabled
#   sakshi_omega  — full pipeline including Phase 2 Omega
#
# Available models: gpt-5.5 | claude-sonnet-4-6 | llama-3.3-70b | qwen-3.5-9b
# Results saved to: results/{model_id}/{mode}.json
#
# Features:
#   - Progress autosave every 10 prompts (crash-safe)
#   - Resume from checkpoint if output file already exists
#   - Per-class accuracy and intervention tracking
#   - Summary statistics printed at completion
# =============================================================================

import argparse
import sys
import os
import json
import time
import traceback
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src")))

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser(description="Sakshi-Protocol V3 benchmark runner")
parser.add_argument("--mode", choices=["baseline","sakshi","sakshi_omega"], default="sakshi_omega")
parser.add_argument("--model", default="gpt-5.5",
    help="Model identifier. Options: gpt-5.5 | claude-sonnet-4-6 | llama-3.3-70b | qwen-3.5-9b")
parser.add_argument("--prompts", default="prompts_500.json")
parser.add_argument("--output-dir", default=None)
parser.add_argument("--limit", type=int, default=None, help="Limit prompts for testing")
parser.add_argument("--no-resume", action="store_true", default=False)
args = parser.parse_args()

MODE         = args.mode
MODEL_ID     = args.model
PROMPTS_FILE = args.prompts
RESUME       = not args.no_resume
LIMIT        = args.limit
OUTPUT_DIR   = args.output_dir or os.path.join(SCRIPT_DIR, "..", "results", MODEL_ID)
AUTOSAVE_EVERY = 10

os.environ["MODEL_ID"] = MODEL_ID

print("=" * 65)
print("  Sakshi-Protocol V3 — Benchmark Runner")
print("=" * 65)
print(f"  Mode:    {MODE}")
print(f"  Model:   {MODEL_ID}")
print(f"  Prompts: {PROMPTS_FILE}")
print(f"  Output:  {OUTPUT_DIR}")
print(f"  Resume:  {RESUME}")
if LIMIT:
    print(f"  Limit:   {LIMIT} prompts")
print()

# =============================================================================
# PIPELINE SETUP
# =============================================================================

from sakshi.models import get_model_client
from sakshi.pipeline import SakshiPipeline

client = get_model_client(MODEL_ID)

if MODE in ("baseline", "sakshi"):
    pipeline = SakshiPipeline(client, omega_enabled=False)
else:
    pipeline = SakshiPipeline(client, omega_enabled=True)
    # For omega mode, use model-specific omega thresholds if available.
    # e.g. llama-3.3-70b-omega prevents omega from running on
    # non-hallucination prompts, protecting accuracy.
    from sakshi.controller import MODEL_THRESHOLDS as _MT, _normalise_model_id as _norm
    omega_key = _norm(MODEL_ID) + "-omega"
    if omega_key in _MT:
        pipeline._model_id = omega_key

print(f"Pipeline ready: {MODE} | omega={pipeline.omega_enabled}")

# =============================================================================
# LOAD PROMPTS
# =============================================================================

prompts_path = PROMPTS_FILE if os.path.isabs(PROMPTS_FILE) else os.path.join(SCRIPT_DIR, PROMPTS_FILE)

with open(prompts_path, encoding="utf-8") as f:
    prompts = json.load(f)

if LIMIT:
    prompts = prompts[:LIMIT]

print(f"Loaded {len(prompts)} prompts\n")

# =============================================================================
# RESUME LOGIC
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, f"{MODE}.json")

results       = []
completed_ids = set()

if RESUME and os.path.exists(output_file):
    try:
        with open(output_file, encoding="utf-8") as f:
            results = json.load(f)
        # Only skip records with valid output — ERROR records get rerun
        valid         = [r for r in results if r.get("output") not in (None, "ERROR") and not r.get("error")]
        invalid       = [r for r in results if r.get("output") in (None, "ERROR") or r.get("error")]
        results       = valid
        completed_ids = {r["id"] for r in valid}
        print(f"Resuming: {len(completed_ids)} valid prompts already done")
        if invalid:
            print(f"Rerunning: {len(invalid)} ERROR records from previous run")
        print()
    except Exception as e:
        print(f"Could not load existing results ({e}), starting fresh\n")
        results, completed_ids = [], set()

prompts_to_run = [p for p in prompts if p["id"] not in completed_ids]
print(f"Prompts to run: {len(prompts_to_run)} / {len(prompts)}\n")

# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def _normalise(text: str) -> str:
    """
    Normalise text for comparison:
        - lowercase
        - remove commas (1,024 → 1024)
        - remove accents (García → garcia)
        - strip punctuation and extra whitespace
    """
    import unicodedata
    import re
    # Lowercase
    text = text.lower()
    # Remove commas from numbers (1,024 → 1024; 299,792,458 → 299792458)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    # Decompose unicode accents and strip combining characters
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    # Remove markdown bold/italic markers
    text = re.sub(r'\*+', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Word/digit equivalences for common answer forms
_WORD_TO_NUM = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4",
    "five":"5","six":"6","seven":"7","eight":"8","nine":"9",
    "ten":"10","eleven":"11","twelve":"12","thirteen":"13",
    "fourteen":"14","fifteen":"15","sixteen":"16","twenty":"20",
}
_NUM_TO_WORD = {v: k for k, v in _WORD_TO_NUM.items()}

def evaluate_correctness(output, ground_truth):
    if not ground_truth or not ground_truth.strip():
        return None
    if not output:
        return 0

    out_norm = _normalise(output)
    gt_norm  = _normalise(ground_truth)

    # Direct substring match after normalisation
    if gt_norm in out_norm:
        return 1

    # Word/digit equivalence check (four ↔ 4)
    gt_words = gt_norm.split()
    for i, w in enumerate(gt_words):
        if w in _WORD_TO_NUM:
            gt_words[i] = _WORD_TO_NUM[w]
        elif w in _NUM_TO_WORD:
            gt_words[i] = _NUM_TO_WORD[w]
    gt_converted = " ".join(gt_words)
    if gt_converted in out_norm:
        return 1

    # Multi-word answer: check if enough key words are present
    answer_words = gt_norm.split()
    if len(answer_words) > 1:
        matches = sum(1 for w in answer_words if w in out_norm)
        if matches >= len(answer_words) * 0.6:
            return 1

    return 0


def format_line(item_id, decision, d_val, pre_gen, correct, elapsed):
    d_str    = f"{d_val:.3f}" if d_val is not None else "N/A"
    corr_str = f"  correct={correct}" if correct is not None else ""
    risk_str = ""
    if pre_gen:
        ep = pre_gen.get("epistemic_profile", {})
        risk_str = f"  risk={ep.get('risk_level','?')}({ep.get('risk_score',0):.2f})"
    return f"  [{item_id}] {decision}  D={d_str}{risk_str}{corr_str}  ({elapsed:.1f}s)"

# =============================================================================
# MAIN LOOP
# =============================================================================

since_save = 0

for item in prompts_to_run:
    item_id     = item["id"]
    prompt      = item["prompt"]
    prompt_type = item.get("type", "")
    gt          = item.get("answer", "")
    domain      = item.get("domain", "")
    hall_class  = item.get("class", "")
    t0          = time.time()

    try:
        if MODE == "baseline":
            output     = pipeline.generator.generate(prompt)
            state      = None
            distortion = None
            distortion_pre_grounding = None
            decision   = "accept"
            intervened = grounded = False
            pre_gen    = None
            trajectory = abstention_result = omega_result = None

        else:
            (output, state, distortion, distortion_pre_grounding,
             decision, intervened, grounded, pre_gen,
             trajectory, abstention_result, omega_result) = pipeline.run(
                prompt, prompt_type=prompt_type
            )

        correct = evaluate_correctness(output, gt)
        elapsed = time.time() - t0
        d_val   = distortion["D_combined"] if isinstance(distortion, dict) else distortion

        print(format_line(item_id, decision, d_val, pre_gen, correct, elapsed))

        # Effective decision — distinguishes grounded accepts from ungrounded
        # sakshi_omega: grounded_accept = retrieved + accepted
        # sakshi:       all accepts are ungrounded (omega_enabled=False)
        # This is the key differentiator between modes in analysis.
        if decision == "accept" and grounded:
            effective_decision = "grounded_accept"
        else:
            effective_decision = decision

        results.append({
            "id":            item_id,
            "model":         MODEL_ID,
            "mode":          MODE,
            "type":          prompt_type,
            "domain":        domain,
            "class":         hall_class,
            "prompt":        prompt,
            "output":        output,
            "ground_truth":  gt,
            "correct":       correct,
            "elapsed_s":     round(elapsed, 2),
            "state":         state,
            "D_combined":    d_val,
            "D_factual":     distortion["D_factual"]   if isinstance(distortion, dict) else None,
            "D_reasoning":   distortion["D_reasoning"] if isinstance(distortion, dict) else None,
            "D_pre_grounding": distortion_pre_grounding["D_combined"] if isinstance(distortion_pre_grounding, dict) else None,
            "decision":      decision,
            "effective_decision": effective_decision,
            "intervened":    intervened,
            "grounded":      grounded,
            "pre_gen":       pre_gen,
            "trajectory":    trajectory.to_dict() if trajectory else None,
            "abstention":    abstention_result.to_dict() if abstention_result else None,
            "omega":         omega_result,
        })

        # Inter-call delay for rate-limited providers
        # GPT-5.5 has aggressive rate limits — needs longer delay
        if MODEL_ID in ("gpt-5.5", "gpt-4o", "gpt-4o-mini"):
            import time as _t
            _t.sleep(3.0)
        elif MODEL_ID in ("llama-3.3-70b", "llama-3.1-8b", "qwen-3.5-9b", "qwen-2.5-7b"):
            import time as _t
            _t.sleep(1.5)
        elif MODEL_ID not in ("claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"):
            import time as _t
            _t.sleep(1.5)

        since_save += 1
        if since_save >= AUTOSAVE_EVERY:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            since_save = 0

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{item_id}] ERROR ({elapsed:.1f}s): {e}")
        traceback.print_exc()
        results.append({
            "id": item_id, "model": MODEL_ID, "mode": MODE,
            "type": prompt_type, "domain": domain, "class": hall_class,
            "prompt": prompt, "output": None, "correct": None,
            "error": str(e),
        })
        since_save += 1

# =============================================================================
# FINAL SAVE
# =============================================================================

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved to {output_file}")

# =============================================================================
# SUMMARY
# =============================================================================

print()
print("=" * 65)
print(f"  SUMMARY — {MODE.upper()} | {MODEL_ID}")
print("=" * 65)

total     = len(results)
errors    = sum(1 for r in results if r.get("error") or r.get("output") in (None, "ERROR"))
completed = total - errors

print(f"  Total: {total}  Completed: {completed}  Errors: {errors}")

if completed > 0:

    # Accuracy by type
    by_type = defaultdict(lambda: {"c": 0, "n": 0})
    for r in results:
        if r.get("correct") is not None and not r.get("error"):
            by_type[r.get("type","?")]["n"] += 1
            by_type[r.get("type","?")]["c"] += r["correct"]
    print()
    print("  Accuracy by type:")
    for t, v in sorted(by_type.items()):
        acc = v["c"] / v["n"] if v["n"] else 0
        print(f"    {t:20s}  {v['c']}/{v['n']}  ({acc:.1%})")

    # Decision distribution
    decisions = defaultdict(int)
    for r in results:
        if not r.get("error"):
            decisions[r.get("decision","?")]  += 1
    print()
    print("  Decisions:")
    for d, c in sorted(decisions.items(), key=lambda x: -x[1]):
        print(f"    {d:15s} {c:4d}  ({c/completed:.1%})")

    # Rates
    iv = sum(1 for r in results if r.get("intervened") and not r.get("error"))
    gr = sum(1 for r in results if r.get("grounded")   and not r.get("error"))
    ab = sum(1 for r in results if r.get("decision") == "abstain" and not r.get("error"))
    ga = sum(1 for r in results if r.get("effective_decision") == "grounded_accept" and not r.get("error"))
    print()
    print(f"  Intervention:     {iv}/{completed} ({iv/completed:.1%})")
    print(f"  Grounding:        {gr}/{completed} ({gr/completed:.1%})")
    print(f"  Abstention:       {ab}/{completed} ({ab/completed:.1%})")
    print(f"  Grounded accepts: {ga}/{completed} ({ga/completed:.1%})  [sakshi_omega differentiator]")

    # Distortion stats
    dv = [r["D_combined"] for r in results if r.get("D_combined") is not None]
    if dv:
        print()
        print(f"  D_combined  mean={sum(dv)/len(dv):.4f}  min={min(dv):.4f}  max={max(dv):.4f}")

    # Hallucination class breakdown
    cd = defaultdict(lambda: defaultdict(int))
    for r in results:
        if r.get("type") == "hallucination" and not r.get("error"):
            cd[r.get("class","?")][r.get("decision","?")] += 1
    if cd:
        print()
        print("  Hallucination class decisions:")
        for cls in sorted(cd.keys()):
            parts = ", ".join(f"{k}={v}" for k, v in sorted(cd[cls].items()))
            print(f"    Class {cls} (n={sum(cd[cls].values())}): {parts}")

    # Pre-gen risk
    rk = defaultdict(int)
    for r in results:
        if r.get("pre_gen") and not r.get("error"):
            ep = r["pre_gen"].get("epistemic_profile", {})
            rk[ep.get("risk_level","?")] += 1
    if rk:
        print()
        print("  Pre-gen risk:")
        for level, count in sorted(rk.items(), key=lambda x: -x[1]):
            print(f"    {level:10s} {count}")

print()
print(f"  Done. {completed}/{total} successful.")
print("=" * 65)
