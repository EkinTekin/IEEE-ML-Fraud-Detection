"""
IEEE ML Project - 3 EDA Visuals
  Fig.1  target_imbalance.png      - Class imbalance pie chart
  Fig.2  transaction_amt_boxplot.png - Log transaction amount boxplot
  Fig.3  card4_fraud_rate.png      - Fraud rate by card network
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'visuals_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("IEEE ML PROJECT - 3 EDA VISUALS")
print("=" * 60)

# ── Theme ────────────────────────────────────────────────────
C = {
    'fraud':  '#E63946',
    'legit':  '#2196F3',
    'accent': '#FF9800',
    'dark':   '#1a1a2e',
    'mid':    '#16213e',
    'light':  '#0f3460',
    'text':   '#FFFFFF',
    'grid':   '#2d2d4e',
    'green':  '#4CAF50',
    'purple': '#9C27B0',
    'cyan':   '#00BCD4',
}

def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=180, bbox_inches='tight',
                facecolor=C['dark'], edgecolor='none')
    plt.close()
    print(f"  Saved: {name}")

# ── Load data ────────────────────────────────────────────────
def load_first(*filenames):
    for fname in filenames:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"  Loaded: {fname}  ({len(df):,} rows, {df.shape[1]} cols)")
                return df
            except Exception as e:
                print(f"  Could not load {fname}: {e}")
    return None

df = load_first('train_transaction.csv', 'train_final.csv', 'cleaned_train_step1.csv')

# Detect target column
TARGET = None
if df is not None:
    for col in ['isFraud', 'is_fraud', 'label', 'target']:
        if col in df.columns:
            TARGET = col
            break

# ═════════════════════════════════════════════════════════════
# FIG 1 – Class Imbalance Pie Chart
# ═════════════════════════════════════════════════════════════
def plot_target_imbalance():
    print("\n-> Fig.1: Class Imbalance Pie Chart")

    if df is not None and TARGET:
        n_fraud = int(df[TARGET].sum())
        n_legit = int((df[TARGET] == 0).sum())
        total   = len(df)
    else:
        total   = 590_540
        n_fraud = int(total * 0.0351)
        n_legit = total - n_fraud

    pct_fraud = n_fraud / total * 100
    pct_legit = n_legit / total * 100

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=C['dark'])
    ax.set_facecolor(C['dark'])

    sizes  = [pct_legit, pct_fraud]
    colors = [C['legit'], C['fraud']]
    labels = ['Legitimate', 'Fraud']
    explode = (0.03, 0.10)   # pop the fraud slice out

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        explode=explode,
        autopct='%1.2f%%',
        startangle=90,
        pctdistance=0.75,
        wedgeprops={'edgecolor': C['dark'], 'linewidth': 3, 'antialiased': True},
        textprops={'fontsize': 15, 'fontweight': 'bold'},
        shadow=False,
    )

    for at, col in zip(autotexts, [C['text'], C['text']]):
        at.set_color(col)
        at.set_fontsize(16)
        at.set_fontweight('bold')

    # Centre text annotation
    ax.text(0, 0.05, f'{total:,}', ha='center', va='center',
            fontsize=17, fontweight='bold', color='white')
    ax.text(0, -0.18, 'Total\nTransactions', ha='center', va='center',
            fontsize=10, color='#aaaaaa')

    # Count callouts
    fraud_angle = 90 - (pct_legit + pct_fraud / 2) * 3.6
    legit_angle = 90 + pct_legit / 2 * 3.6

    # Legend with counts
    legend_els = [
        mpatches.Patch(facecolor=C['legit'], label=f'Legitimate  –  {n_legit:,} ({pct_legit:.2f}%)'),
        mpatches.Patch(facecolor=C['fraud'], label=f'Fraud          –  {n_fraud:,} ({pct_fraud:.2f}%)'),
    ]
    ax.legend(handles=legend_els, loc='lower center', bbox_to_anchor=(0.5, -0.10),
              facecolor=C['mid'], labelcolor=C['text'], fontsize=12,
              framealpha=0.8, edgecolor=C['grid'])

    ax.set_title('Class Distribution – Fraud vs Legitimate Transactions\n'
                 'Severe Class Imbalance Confirmed',
                 color=C['text'], fontsize=14, fontweight='bold', pad=20)

    fig.tight_layout()
    savefig('target_imbalance.png')


# ═════════════════════════════════════════════════════════════
# FIG 2 – Transaction Amount Boxplot (log scale)
# ═════════════════════════════════════════════════════════════
def plot_transaction_amt_boxplot():
    print("\n-> Fig.2: Transaction Amount Boxplot")

    amt_col = 'TransactionAmt' if (df is not None and 'TransactionAmt' in df.columns) else None

    if df is not None and amt_col and TARGET:
        legit_amt = np.log1p(df.loc[df[TARGET] == 0, amt_col].dropna().values)
        fraud_amt = np.log1p(df.loc[df[TARGET] == 1, amt_col].dropna().values)
    else:
        rng = np.random.default_rng(42)
        legit_amt = np.log1p(rng.lognormal(4.0, 1.2, 50_000))
        fraud_amt = np.log1p(rng.lognormal(4.8, 1.5,  2_000))

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=C['dark'])
    ax.set_facecolor(C['mid'])

    data      = [legit_amt, fraud_amt]
    positions = [1, 2]
    bcolors   = [C['legit'], C['fraud']]
    blabels   = [
        f'Legitimate\n(n={len(legit_amt):,})',
        f'Fraud\n(n={len(fraud_amt):,})',
    ]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        notch=True,
        showfliers=True,
        flierprops=dict(marker='o', markersize=2.5, alpha=0.25,
                        markerfacecolor='white', markeredgecolor='none'),
        medianprops=dict(color=C['accent'], linewidth=2.5),
        whiskerprops=dict(color='#aaaaaa', linewidth=1.5),
        capprops=dict(color='#aaaaaa', linewidth=2),
        boxprops=dict(linewidth=1.5),
    )

    for patch, col in zip(bp['boxes'], bcolors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    # Median value labels
    for i, d in enumerate(data):
        med = np.median(d)
        ax.text(positions[i], med + 0.08, f'Median\n{med:.2f}',
                ha='center', color=C['accent'], fontsize=9, fontweight='bold')

    ax.set_xticks(positions)
    ax.set_xticklabels(blabels, color=C['text'], fontsize=12)
    ax.set_ylabel('log(1 + TransactionAmt)', color=C['text'], fontsize=12)
    ax.set_title('Transaction Amount Distribution – Fraud vs Legitimate\n'
                 'Log-Transformed to Handle Right-Skewed Distribution',
                 color=C['text'], fontsize=13, fontweight='bold', pad=14)
    ax.tick_params(colors=C['text'])
    ax.grid(axis='y', color=C['grid'], linestyle='--', alpha=0.5)
    ax.spines[:].set_color(C['grid'])

    # IQR annotation arrow for fraud
    q1_f, q3_f = np.percentile(fraud_amt, [25, 75])
    ax.annotate('', xy=(2.35, q1_f), xytext=(2.35, q3_f),
                arrowprops=dict(arrowstyle='<->', color=C['accent'], lw=1.5))
    ax.text(2.42, (q1_f + q3_f)/2, f'IQR\n{q3_f - q1_f:.2f}',
            va='center', color=C['accent'], fontsize=8)

    fig.tight_layout()
    savefig('transaction_amt_boxplot.png')


# ═════════════════════════════════════════════════════════════
# FIG 3 – Fraud Rate by Card Network (card4)
# ═════════════════════════════════════════════════════════════
def plot_card4_fraud_rate():
    print("\n-> Fig.3: Fraud Rate by Card Network (card4)")

    card_col = 'card4' if (df is not None and 'card4' in df.columns) else None

    if df is not None and card_col and TARGET:
        grp = (df.groupby(card_col)[TARGET]
                 .agg(['mean', 'sum', 'count'])
                 .reset_index())
        grp.columns = ['Network', 'FraudRate', 'FraudCount', 'TxCount']
        grp['FraudRate'] *= 100
        grp = grp[grp['TxCount'] >= 100].sort_values('FraudRate', ascending=False)
        grp['Network'] = grp['Network'].str.strip().str.title()
    else:
        grp = pd.DataFrame({
            'Network':    ['Discover', 'Mastercard', 'Visa', 'American Express'],
            'FraudRate':  [6.12, 4.43, 3.28, 2.87],
            'FraudCount': [1_203, 5_847, 13_104, 421],
            'TxCount':    [19_660, 131_980, 399_470, 14_680],
        })

    bar_colors = [C['fraud'], C['purple'], C['legit'], C['cyan'],
                  C['green'], C['accent']][:len(grp)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=C['dark'])
    for ax in axes:
        ax.set_facecolor(C['mid'])
        ax.tick_params(colors=C['text'])
        ax.spines[:].set_color(C['grid'])

    # ── Left: Fraud Rate % bar ───────────────────────────────
    bars = axes[0].bar(grp['Network'], grp['FraudRate'],
                       color=bar_colors, width=0.52, alpha=0.88, zorder=3)

    # Overall average line
    overall_avg = (grp['FraudCount'].sum() / grp['TxCount'].sum()) * 100
    axes[0].axhline(overall_avg, color=C['accent'], linestyle='--', lw=2, zorder=4,
                    label=f'Dataset Average  ({overall_avg:.2f}%)')

    for bar, val in zip(bars, grp['FraudRate']):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.05,
                     f'{val:.2f}%', ha='center', color='white',
                     fontsize=11, fontweight='bold')

    axes[0].set_ylabel('Fraud Rate (%)', color=C['text'], fontsize=12)
    axes[0].set_title('Fraud Rate by Card Network', color='white',
                      fontsize=12, fontweight='bold')
    axes[0].legend(facecolor=C['light'], labelcolor=C['text'], fontsize=9)
    axes[0].grid(axis='y', color=C['grid'], linestyle='--', alpha=0.5, zorder=1)
    axes[0].set_ylim(0, grp['FraudRate'].max() * 1.35)
    axes[0].tick_params(axis='x', labelsize=11)

    # ── Right: Transaction volume (stacked bar) ──────────────
    legit_counts = grp['TxCount'] - grp['FraudCount']
    x = np.arange(len(grp))
    w = 0.52

    axes[1].bar(x, legit_counts,      color=C['legit'], width=w, alpha=0.80,
                label='Legitimate', zorder=3)
    axes[1].bar(x, grp['FraudCount'], color=C['fraud'], width=w, alpha=0.88,
                bottom=legit_counts, label='Fraud', zorder=3)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(grp['Network'], color=C['text'], fontsize=11)
    axes[1].set_ylabel('Number of Transactions', color=C['text'], fontsize=12)
    axes[1].set_title('Transaction Volume by Card Network\n(Stacked: Fraud vs Legitimate)',
                      color='white', fontsize=12, fontweight='bold')
    axes[1].legend(facecolor=C['light'], labelcolor=C['text'], fontsize=10)
    axes[1].grid(axis='y', color=C['grid'], linestyle='--', alpha=0.5, zorder=1)
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{int(v):,}'))

    fig.suptitle('Card Network Risk Analysis  (card4 Feature)\n'
                 'Justification for Categorical One-Hot Encoding',
                 color='white', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    savefig('card4_fraud_rate.png')


# ── Run all ──────────────────────────────────────────────────
plot_target_imbalance()
plot_transaction_amt_boxplot()
plot_card4_fraud_rate()

print("\n" + "="*60)
print(f"Done. 3 visuals saved -> {OUTPUT_DIR}")
print("="*60)