#!/usr/bin/env python3
"""
Data Mining Lab 3 — Decision Tree Analysis
Datasets: German Credit (binary) & Waveform (3-class)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import io
import os
import json

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, f1_score

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE = "/Users/abdullah/Documents/Data_Mining/Labs/Lab 3"
OUT  = f"{BASE}/output"
os.makedirs(OUT, exist_ok=True)

# ─── Load Datasets ─────────────────────────────────────────────────────────────
# German Credit (space-separated numeric, label in last col: 1=good,2=bad → 0/1)
german_raw = pd.read_csv(
    f"{BASE}/statlog+german+credit+data/german.data-numeric",
    sep=r'\s+', header=None)
X_german = german_raw.iloc[:, :-1].values.astype(float)
y_german  = (german_raw.iloc[:, -1].values - 1)   # remap 1,2 → 0,1

# Waveform (comma-separated, last col: 0/1/2, file is .Z compressed)
proc = subprocess.run(
    ['uncompress', '-c',
     f"{BASE}/waveform+database+generator+version+1/waveform.data.Z"],
    capture_output=True, check=True)
waveform_df = pd.read_csv(io.BytesIO(proc.stdout), header=None)
X_wave = waveform_df.iloc[:, :-1].values.astype(float)
y_wave = waveform_df.iloc[:, -1].values

print(f"German Credit : X={X_german.shape}, classes={np.unique(y_german)}")
print(f"Waveform      : X={X_wave.shape},  classes={np.unique(y_wave)}")

DATASETS = [
    ('German Credit', 'german', X_german, y_german),
    ('Waveform',      'wave',   X_wave,   y_wave),
]

# ─── Helper utilities ──────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, f1


def holdout_experiment(X, y, criterion, n_trials=5, test_size=0.1, seed=0):
    rows = []
    for i in range(n_trials):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed + i, stratify=None)
        clf = DecisionTreeClassifier(criterion=criterion, random_state=seed + i)
        clf.fit(X_tr, y_tr)
        acc, prec, f1 = metrics(y_te, clf.predict(X_te))
        rows.append({'Label': f'Trial {i+1}', 'Accuracy': acc,
                     'Precision': prec, 'F-measure': f1})
    df = pd.DataFrame(rows)
    avg = df[['Accuracy','Precision','F-measure']].mean()
    df = pd.concat([df, pd.DataFrame([{
        'Label': 'Average', **avg.to_dict()}])], ignore_index=True)
    return df


def cv_experiment(X, y, criterion, n_folds=10, seed=0):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for i, (tr_idx, te_idx) in enumerate(kf.split(X)):
        clf = DecisionTreeClassifier(criterion=criterion, random_state=seed)
        clf.fit(X[tr_idx], y[tr_idx])
        acc, prec, f1 = metrics(y[te_idx], clf.predict(X[te_idx]))
        rows.append({'Label': f'Fold {i+1}', 'Accuracy': acc,
                     'Precision': prec, 'F-measure': f1})
    df = pd.DataFrame(rows)
    avg = df[['Accuracy','Precision','F-measure']].mean()
    df = pd.concat([df, pd.DataFrame([{
        'Label': 'Average', **avg.to_dict()}])], ignore_index=True)
    return df


def bar_chart(df, title, filepath):
    """Grouped bar chart: Accuracy / Precision / F-measure per trial or fold."""
    labels = df['Label'].tolist()
    n = len(labels)
    x = np.arange(n)
    w = 0.26

    fig, ax = plt.subplots(figsize=(max(9, n * 0.75), 4.5))
    b1 = ax.bar(x - w, df['Accuracy'],  w, label='Accuracy',  color='#2B7BB9')
    b2 = ax.bar(x,     df['Precision'], w, label='Precision', color='#E87722')
    b3 = ax.bar(x + w, df['F-measure'], w, label='F-measure', color='#3BAA34')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.axvline(n - 1.5, color='gray', linestyle='--', lw=0.8, alpha=0.7)

    # Annotate average group
    for b in [b1, b2, b3]:
        bar = list(b)[-1]
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def depth_experiment(X, y, criterion, depths, seed=0):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.1, random_state=seed)
    train_acc, test_acc = [], []
    for d in depths:
        clf = DecisionTreeClassifier(criterion=criterion,
                                     max_depth=d, random_state=seed)
        clf.fit(X_tr, y_tr)
        train_acc.append(accuracy_score(y_tr, clf.predict(X_tr)))
        test_acc.append(accuracy_score(y_te, clf.predict(X_te)))
    return train_acc, test_acc


def depth_plot(depths, train_acc, test_acc, title, filepath):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(depths, train_acc, 'o-',  color='#2B7BB9', label='Train Accuracy', lw=1.8)
    ax.plot(depths, test_acc,  's--', color='#E87722', label='Test Accuracy',  lw=1.8)
    ax.set_xlabel('Max Tree Depth')
    ax.set_ylabel('Accuracy')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xticks(depths)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3)
    # Mark best test depth
    best_d = depths[int(np.argmax(test_acc))]
    best_a = max(test_acc)
    ax.axvline(best_d, color='green', linestyle=':', lw=1.2, alpha=0.7,
               label=f'Best test depth={best_d}')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

# ─── Run all experiments ───────────────────────────────────────────────────────
all_results = {}   # store DataFrames for LaTeX table generation

for ds_name, ds_key, X, y in DATASETS:
    print(f"\n{'='*60}")
    print(f"  Dataset: {ds_name}")
    print(f"{'='*60}")
    all_results[ds_key] = {}

    for part_label, criterion in [('gini', 'gini'), ('entropy', 'entropy')]:
        print(f"\n  Criterion: {criterion.upper()}")

        # --- Holdout ---
        ho = holdout_experiment(X, y, criterion)
        print(f"\n  [Holdout]\n{ho.to_string(index=False)}")
        bar_chart(
            ho,
            f"{ds_name} | Holdout (5×90/10) | {criterion.title()}",
            f"{OUT}/{ds_key}_holdout_{criterion}.png")

        # --- 10-Fold CV ---
        cv = cv_experiment(X, y, criterion)
        print(f"\n  [10-Fold CV]\n{cv.to_string(index=False)}")
        bar_chart(
            cv,
            f"{ds_name} | 10-Fold Cross-Validation | {criterion.title()}",
            f"{OUT}/{ds_key}_cv_{criterion}.png")

        all_results[ds_key][criterion] = {'holdout': ho, 'cv': cv}

    # --- Part II comparison: CV accuracy Gini vs Entropy ---
    gini_cv = all_results[ds_key]['gini']['cv']
    entr_cv = all_results[ds_key]['entropy']['cv']

    g_vals = gini_cv[gini_cv['Label'] != 'Average']['Accuracy'].values
    e_vals = entr_cv[entr_cv['Label'] != 'Average']['Accuracy'].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Line plot
    ax = axes[0]
    folds = np.arange(1, 11)
    ax.plot(folds, g_vals, 'o-',  color='#2B7BB9',
            label=f'Gini (avg={g_vals.mean():.3f})')
    ax.plot(folds, e_vals, 's--', color='#E87722',
            label=f'Entropy (avg={e_vals.mean():.3f})')
    ax.axhline(g_vals.mean(), color='#2B7BB9', linestyle=':', alpha=0.5)
    ax.axhline(e_vals.mean(), color='#E87722', linestyle=':', alpha=0.5)
    ax.set_xlabel('Fold'); ax.set_ylabel('Accuracy')
    ax.set_title(f'{ds_name} — CV Accuracy per Fold')
    ax.set_xticks(folds); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Bar of averages
    ax2 = axes[1]
    bars = ax2.bar(['Gini', 'Entropy'], [g_vals.mean(), e_vals.mean()],
                   color=['#2B7BB9', '#E87722'], width=0.45)
    for b in bars:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                 f'{b.get_height():.4f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')
    ax2.set_ylabel('Average CV Accuracy')
    ax2.set_title(f'{ds_name} — Average CV Accuracy: Gini vs Entropy')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle(f'{ds_name}: Part I (Gini) vs Part II (Entropy)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{ds_key}_compare_cv.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Comparison figure saved: {ds_key}_compare_cv.png")

    # --- Part III: depth effect ---
    depths = list(range(2, 51, 3))   # 2, 5, 8, 11, ..., 50
    tr_acc, te_acc = depth_experiment(X, y, 'gini', depths)

    print(f"\n  [Part III — Depth Effect | {ds_name}]")
    for d, tr, te in zip(depths, tr_acc, te_acc):
        print(f"    depth={d:2d}  train={tr:.4f}  test={te:.4f}")

    depth_plot(depths, tr_acc, te_acc,
               f"{ds_name} — Effect of Tree Depth on Accuracy (Gini, 90/10 Holdout)",
               f"{OUT}/{ds_key}_depth_effect.png")

    all_results[ds_key]['depths']     = depths
    all_results[ds_key]['train_accs'] = tr_acc
    all_results[ds_key]['test_accs']  = te_acc

# ─── Persist numerical results for LaTeX ──────────────────────────────────────
def df_to_dict(df):
    return df.to_dict(orient='records')

serialisable = {}
for ds_key in ('german', 'wave'):
    serialisable[ds_key] = {}
    for criterion in ('gini', 'entropy'):
        serialisable[ds_key][criterion] = {
            'holdout': df_to_dict(all_results[ds_key][criterion]['holdout']),
            'cv':      df_to_dict(all_results[ds_key][criterion]['cv']),
        }
    serialisable[ds_key]['depths']     = all_results[ds_key]['depths']
    serialisable[ds_key]['train_accs'] = all_results[ds_key]['train_accs']
    serialisable[ds_key]['test_accs']  = all_results[ds_key]['test_accs']

with open(f"{OUT}/results.json", 'w') as f:
    json.dump(serialisable, f, indent=2)

print(f"\n{'='*60}")
print("All experiments done. Results in:", OUT)
print(f"{'='*60}")
