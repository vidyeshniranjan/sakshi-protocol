"""
metrics.py — Sakshi-Protocol V3.0 results analysis
Computes all paper metrics across 3 models × 3 modes.
Run from experiments/ directory.

Usage:
    python metrics.py
    python metrics.py --save   # saves tables to metrics_output/
"""

import json
import os
import re
import unicodedata
import argparse
from collections import defaultdict
from pathlib import Path

# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def _normalise(t):
    t = t.lower()
    t = re.sub(r'(\d),(\d)', r'\1\2', t)
    t = unicodedata.normalize('NFD', t)
    t = ''.join(c for c in t if unicodedata.category(c) != 'Mn')
    return re.sub(r'\s+', ' ', t).strip()

_W2N = {'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
        'six':'6','seven':'7','eight':'8','nine':'9','ten':'10','twelve':'12'}
_N2W = {v: k for k, v in _W2N.items()}

def evaluate(output, gt):
    if not gt or not gt.strip(): return None
    if not output: return 0
    on = _normalise(output)
    gn = _normalise(gt)
    if gn in on: return 1
    gw = gn.split()
    for i, w in enumerate(gw):
        if w in _W2N: gw[i] = _W2N[w]
        elif w in _N2W: gw[i] = _N2W[w]
    if ' '.join(gw) in on: return 1
    aw = gn.split()
    if len(aw) > 1 and sum(1 for w in aw if w in on) >= len(aw) * 0.6:
        return 1
    return 0

def is_genuine_grounding(output):
    """Returns True if the output used retrieved context meaningfully."""
    if not output:
        return False
    out = output.lower()
    irrelevant_markers = [
        'verified context provided is', 'context provided is not',
        'context does not', 'context provided does not',
        'not relevant to', 'does not specifically address',
        'does not contain specific', 'context provided is entirely',
        'entirely unrelated', 'entirely irrelevant',
    ]
    return not any(m in out for m in irrelevant_markers)

# =============================================================================
# RESULTS LOADER
# =============================================================================

RESULTS_BASE = Path('../results')

MODELS = {
    'claude-sonnet-4-6': 'Claude Sonnet 4.6',
    'llama-3.3-70b':     'Llama 3.3 70B',
    'qwen-3.5-9b':       'Qwen 3.5 9B',
}

MODES = ['baseline', 'sakshi', 'sakshi_omega']


def load_results(model_id, mode):
    path = RESULTS_BASE / model_id / f'{mode}.json'
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def compute_metrics(data):
    """Compute all metrics for one model/mode run."""
    if data is None:
        return None

    completed = [r for r in data
                 if r.get('output') not in (None, 'ERROR') and not r.get('error')]
    errors = [r for r in data
              if r.get('output') in (None, 'ERROR') or r.get('error')]

    m = {
        'n':         len(data),
        'completed': len(completed),
        'errors':    len(errors),
    }

    # Accuracy
    for ptype in ['factual', 'reasoning']:
        labelled = [r for r in completed
                    if r.get('type') == ptype and r.get('ground_truth', '').strip()]
        correct  = sum(1 for r in labelled
                       if evaluate(r.get('output', ''), r.get('ground_truth', '')) == 1)
        m[f'acc_{ptype}']       = correct / len(labelled) if labelled else None
        m[f'acc_{ptype}_n']     = len(labelled)
        m[f'acc_{ptype}_correct'] = correct

    # Overall accuracy
    all_labelled = [r for r in completed if r.get('ground_truth', '').strip()]
    all_correct  = sum(1 for r in all_labelled
                       if evaluate(r.get('output', ''), r.get('ground_truth', '')) == 1)
    m['acc_overall'] = all_correct / len(all_labelled) if all_labelled else None

    # Decisions
    decisions = defaultdict(int)
    for r in completed:
        decisions[r.get('effective_decision', r.get('decision', '?'))] += 1
    m['decisions'] = dict(decisions)
    m['n_accept']          = decisions.get('accept', 0)
    m['n_abstain']         = decisions.get('abstain', 0)
    m['n_grounded_accept'] = decisions.get('grounded_accept', 0)

    # Rates
    n = len(completed)
    m['abstain_rate']         = m['n_abstain'] / n if n else 0
    m['grounded_accept_rate'] = m['n_grounded_accept'] / n if n else 0
    m['intervention_rate']    = (m['n_abstain'] + m['n_grounded_accept']) / n if n else 0

    # False positives
    fp = sum(1 for r in completed
             if r.get('type') in ('factual', 'reasoning', 'ambiguous')
             and r.get('decision') in ('abstain', 'retrieve'))
    m['false_positives'] = fp

    # Hallucination class breakdown
    for cls in ['A', 'B', 'C']:
        cd = defaultdict(int)
        for r in completed:
            if r.get('type') == 'hallucination' and r.get('class') == cls:
                cd[r.get('effective_decision', r.get('decision', '?'))] += 1
        total = sum(cd.values())
        m[f'class_{cls}_total']          = total
        m[f'class_{cls}_abstain']        = cd.get('abstain', 0)
        m[f'class_{cls}_accept']         = cd.get('accept', 0)
        m[f'class_{cls}_grounded_accept']= cd.get('grounded_accept', 0)
        m[f'class_{cls}_intervention']   = (cd.get('abstain', 0) +
                                            cd.get('grounded_accept', 0))
        m[f'class_{cls}_intervention_rate'] = (
            m[f'class_{cls}_intervention'] / total if total else 0
        )

    # Grounding (omega only)
    grounded = [r for r in completed if r.get('grounded')]
    genuine  = [r for r in grounded if is_genuine_grounding(r.get('output', ''))]
    m['n_grounded']          = len(grounded)
    m['n_genuine_grounding'] = len(genuine)
    m['genuine_grounding_rate'] = (len(genuine) / len(grounded)
                                   if grounded else 0)

    # Anupalabdhi paths
    paths = defaultdict(int)
    for r in completed:
        p = (r.get('omega') or {}).get('anupalabdhi_path')
        if p:
            paths[p] += 1
    m['anupalabdhi_paths'] = dict(paths)

    # Natural hedges (baseline)
    hall = [r for r in completed if r.get('type') == 'hallucination']
    hedges = [r for r in hall if any(x in r.get('output', '').lower() for x in [
        "i can't", "i cannot", "i'm not aware", "not aware of", "no record",
        "doesn't exist", "does not exist", "cannot verify", "unable to verify",
        "no evidence", "not real", "i don't have", "couldn't verify",
        "couldn't find", "no information", "fabricated", "made up",
    ])]
    m['natural_hedge_rate'] = len(hedges) / len(hall) if hall else 0
    for cls in ['A', 'B', 'C']:
        ch = [r for r in hall if r.get('class') == cls]
        cr = [r for r in hedges if r.get('class') == cls]
        m[f'natural_hedge_class_{cls}'] = len(cr) / len(ch) if ch else 0

    # D_combined stats
    d_vals = [r.get('D_combined', 0) for r in completed
              if r.get('D_combined') is not None]
    if d_vals:
        m['d_mean'] = sum(d_vals) / len(d_vals)
        m['d_min']  = min(d_vals)
        m['d_max']  = max(d_vals)

    return m


# =============================================================================
# MAIN
# =============================================================================

def main(save=False):
    results = {}
    for model_id in MODELS:
        results[model_id] = {}
        for mode in MODES:
            data = load_results(model_id, mode)
            results[model_id][mode] = compute_metrics(data)
            status = 'OK' if data else 'MISSING'
            print(f'  {model_id:20s} {mode:15s} {status}')

    print()
    print_summary_table(results)
    print()
    print_hallucination_table(results)
    print()
    print_accuracy_table(results)
    print()
    print_grounding_table(results)

    if save:
        os.makedirs('metrics_output', exist_ok=True)
        save_results(results)
        print('\nSaved to metrics_output/')


def print_summary_table(results):
    print('=' * 90)
    print('TABLE 1: SUMMARY METRICS ACROSS ALL RUNS')
    print('=' * 90)
    header = f'{"Model":22s} {"Mode":15s} {"Acc(F)":>8} {"Acc(R)":>8} {"Abstain":>9} {"Grounded":>10} {"FP":>5} {"Errors":>7}'
    print(header)
    print('-' * 90)
    for model_id, model_name in MODELS.items():
        for mode in MODES:
            m = results[model_id].get(mode)
            if m is None:
                print(f'  {model_name:20s} {mode:15s} MISSING')
                continue
            acc_f  = f'{m["acc_factual"]*100:.1f}%'   if m.get("acc_factual")  is not None else 'N/A'
            acc_r  = f'{m["acc_reasoning"]*100:.1f}%' if m.get("acc_reasoning") is not None else 'N/A'
            abst   = f'{m["abstain_rate"]*100:.1f}%'
            grnd   = f'{m["grounded_accept_rate"]*100:.1f}%'
            fp     = str(m['false_positives'])
            errors = str(m['errors'])
            print(f'  {model_name:20s} {mode:15s} {acc_f:>8} {acc_r:>8} {abst:>9} {grnd:>10} {fp:>5} {errors:>7}')
        print()


def print_hallucination_table(results):
    print('=' * 90)
    print('TABLE 2: HALLUCINATION INTERVENTION RATES BY CLASS')
    print('=' * 90)
    header = f'{"Model":22s} {"Mode":15s} {"Class A":>10} {"Class B":>10} {"Class C":>10} {"Total Hall":>12}'
    print(header)
    print('-' * 90)
    for model_id, model_name in MODELS.items():
        for mode in MODES:
            m = results[model_id].get(mode)
            if m is None:
                continue
            ca = f'{m["class_A_intervention_rate"]*100:.0f}%'
            cb = f'{m["class_B_intervention_rate"]*100:.0f}%'
            cc = f'{m["class_C_intervention_rate"]*100:.0f}%'
            total_hall = m['class_A_total'] + m['class_B_total'] + m['class_C_total']
            total_int  = (m['class_A_intervention'] + m['class_B_intervention'] +
                         m['class_C_intervention'])
            tot = f'{total_int/total_hall*100:.0f}%' if total_hall else 'N/A'
            print(f'  {model_name:20s} {mode:15s} {ca:>10} {cb:>10} {cc:>10} {tot:>12}')
        print()


def print_accuracy_table(results):
    print('=' * 70)
    print('TABLE 3: ACCURACY - BASELINE vs SAKSHI vs OMEGA')
    print('=' * 70)
    for model_id, model_name in MODELS.items():
        print(f'  {model_name}:')
        for mode in MODES:
            m = results[model_id].get(mode)
            if m is None:
                continue
            acc_f = f'{m["acc_factual"]*100:.1f}%'   if m.get("acc_factual")  is not None else 'N/A'
            acc_r = f'{m["acc_reasoning"]*100:.1f}%' if m.get("acc_reasoning") is not None else 'N/A'
            fp    = m['false_positives']
            print(f'    {mode:15s}  factual={acc_f}  reasoning={acc_r}  FP={fp}')
        print()


def print_grounding_table(results):
    print('=' * 80)
    print('TABLE 4: GROUNDING ANALYSIS (Sakshi+Omega only)')
    print('=' * 80)
    header = f'{"Model":22s} {"Grounded":>10} {"Genuine":>10} {"Genuine%":>10} {"Class A GA":>12} {"Class B GA":>12} {"Class C GA":>12}'
    print(header)
    print('-' * 80)
    for model_id, model_name in MODELS.items():
        m = results[model_id].get('sakshi_omega')
        if m is None:
            print(f'  {model_name:22s} MISSING')
            continue
        gr   = m['n_grounded']
        gen  = m['n_genuine_grounding']
        genp = f'{m["genuine_grounding_rate"]*100:.0f}%'
        ca_ga = m['class_A_grounded_accept']
        cb_ga = m['class_B_grounded_accept']
        cc_ga = m['class_C_grounded_accept']
        print(f'  {model_name:22s} {gr:>10} {gen:>10} {genp:>10} {ca_ga:>12} {cb_ga:>12} {cc_ga:>12}')


def save_results(results):
    """Save computed metrics to JSON for use by plotting scripts."""
    # Convert to serialisable format
    serialisable = {}
    for model_id, modes in results.items():
        serialisable[model_id] = {}
        for mode, m in modes.items():
            if m is not None:
                serialisable[model_id][mode] = m

    with open('metrics_output/metrics.json', 'w') as f:
        json.dump(serialisable, f, indent=2)
    print('  metrics.json saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true',
                        help='Save metrics to metrics_output/')
    args = parser.parse_args()
    main(save=args.save)
