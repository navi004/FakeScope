"""
train_model.py
─────────────────────────────────────────────────────────────
Fake Engagement Detection — Model Training
Dataset : airt-ml/twitter-human-bots (real, public, CC-BY-SA 3.0)
Model   : Random Forest Classifier (sklearn)
─────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib, os, json, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight

os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

DARK_BG = '#0D0F14'; CARD_BG = '#1A1D26'; BORDER = '#2A2D3A'
GREEN = '#4ECCA3'; RED = '#FF4757'; GREY = '#8B8FA8'

# Load
print("Loading processed dataset...")
df = pd.read_csv("data/processed_dataset.csv")
FEATURES = [c for c in df.columns if c != 'label']
X = df[FEATURES]; y = df['label']
print(f"   {len(df):,} records | {len(FEATURES)} features | {y.mean()*100:.1f}% bots")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
cw = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)
cw_dict = {0: cw[0], 1: cw[1]}

print("\nTraining Random Forest (300 trees)...")
model = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=5,
                                min_samples_leaf=2, class_weight=cw_dict, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy"  : round(accuracy_score(y_test, y_pred), 4),
    "precision" : round(precision_score(y_test, y_pred), 4),
    "recall"    : round(recall_score(y_test, y_pred), 4),
    "f1_score"  : round(f1_score(y_test, y_pred), 4),
    "roc_auc"   : round(roc_auc_score(y_test, y_prob), 4),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
metrics['cv_auc_mean'] = round(cv_auc.mean(), 4)
metrics['cv_auc_std']  = round(cv_auc.std(), 4)

print("\n  Results:")
for k, v in metrics.items(): print(f"    {k}: {v}")
with open("outputs/metrics.json", "w") as f: json.dump(metrics, f, indent=2)

def dark_fig(w=7, h=5):
    fig, ax = plt.subplots(figsize=(w,h), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.tick_params(colors=GREY); ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY); ax.title.set_color('white')
    return fig, ax

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = dark_fig(6,5)
ax.imshow(cm, cmap='Blues', alpha=0.85)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Human','Bot'], fontsize=13, color='white')
ax.set_yticklabels(['Human','Bot'], fontsize=13, color='white')
ax.set_xlabel('Predicted', fontsize=13); ax.set_ylabel('Actual', fontsize=13)
ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold', pad=14)
for i in range(2):
    for j in range(2):
        c = 'white' if cm[i,j]>cm.max()/2 else GREY
        ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center', fontsize=18, color=c, fontweight='bold')
plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches='tight'); plt.close()

# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = dark_fig(7,5)
ax.plot(fpr, tpr, color=RED, lw=2.5, label=f'ROC AUC = {metrics["roc_auc"]:.4f}')
ax.fill_between(fpr, tpr, alpha=0.08, color=RED)
ax.plot([0,1],[0,1],'--',lw=1,color=BORDER)
ax.set_xlabel('False Positive Rate',fontsize=12); ax.set_ylabel('True Positive Rate',fontsize=12)
ax.set_title('ROC Curve — Bot Detection', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, framealpha=0.2, facecolor=CARD_BG, labelcolor='white')
ax.grid(alpha=0.15, color=BORDER)
plt.tight_layout(); plt.savefig("outputs/roc_curve.png", dpi=150, bbox_inches='tight'); plt.close()

# Feature Importance
feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
top20 = feat_imp.tail(20)
fig, ax = dark_fig(10,8)
colors = [RED if v > top20.median() else GREEN for v in top20.values]
bars = ax.barh(top20.index, top20.values, color=colors, height=0.7, edgecolor=BORDER, linewidth=0.5)
ax.set_xlabel('Feature Importance (Gini)', fontsize=12)
ax.set_title('Top 20 Behavioural Features\nDriving Bot vs Human Classification', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.15, color=BORDER)
for bar, val in zip(bars, top20.values):
    ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9, color=GREY)
plt.tight_layout(); plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches='tight')
plt.savefig("outputs/shap_bar.png", dpi=150, bbox_inches='tight'); plt.close()

# Distribution plots
top_feats = feat_imp.tail(6).index.tolist()[::-1]
fig, axes = plt.subplots(2, 3, figsize=(14,8), facecolor=DARK_BG)
fig.suptitle('Behavioural Feature Distributions: Bot vs Human', fontsize=14, color='white', fontweight='bold', y=1.01)
df_full = pd.concat([X, y], axis=1)
for ax, feat in zip(axes.flat, top_feats):
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.hist(df_full[df_full['label']==0][feat], bins=40, alpha=0.7, color=GREEN, density=True, label='Human')
    ax.hist(df_full[df_full['label']==1][feat], bins=40, alpha=0.7, color=RED,   density=True, label='Bot')
    ax.set_title(feat.replace('_',' ').title(), fontsize=10, color='white')
    ax.tick_params(colors=GREY, labelsize=8)
    ax.legend(fontsize=8, framealpha=0.2, facecolor=CARD_BG, labelcolor='white')
    ax.grid(alpha=0.1, color=BORDER)
plt.tight_layout(); plt.savefig("outputs/feature_distributions.png", dpi=150, bbox_inches='tight'); plt.close()

joblib.dump(model, "models/rf_model.pkl")
pd.Series(FEATURES).to_csv("models/features.csv", index=False, header=False)
print("\nAll outputs saved. Training complete!")
