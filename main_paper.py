import os
import matplotlib
matplotlib.use('Agg')
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import json

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from config_paper import *
from utils_paper import load_and_preprocess_data
from model_paper import UGCNet
from trainer_paper import Trainer

VAL_WELL_NAMES = ['柴902', '柴909', '柴12', '柴906']
LAYER_NAMES = ['III', 'IV', 'V', 'VI']


def plot_single_well(well_data, pred, expected_value, well_name, save_dir):
    fig, axes = plt.subplots(3, 1, figsize=(15, 16))
    fig.suptitle(f'{well_name} - UGCNet (paper)', fontsize=16, fontweight='bold')
    tvdss = well_data['tvdss']; y_true = well_data['labels']; vm = well_data['valid_mask']

    s1 = axes[0].scatter(range(len(y_true)), -tvdss, c=y_true, cmap='viridis', s=12, alpha=0.8)
    axes[0].set_title('True'); axes[0].set_ylabel('TVDSS (inv)')
    cbar1 = plt.colorbar(s1, ax=axes[0]); cbar1.set_ticks([0, 1, 2, 3]); cbar1.set_ticklabels(LAYER_NAMES)

    s2 = axes[1].scatter(range(len(pred)), -tvdss, c=pred, cmap='viridis', s=12, alpha=0.8)
    axes[1].set_title('Pred'); axes[1].set_ylabel('TVDSS (inv)')
    cbar2 = plt.colorbar(s2, ax=axes[1]); cbar2.set_ticks([0, 1, 2, 3]); cbar2.set_ticklabels(LAYER_NAMES)

    axes[2].plot(expected_value, color='#2C3E50', linewidth=1)
    axes[2].fill_between(range(len(expected_value)), 0, expected_value, color='#2C3E50', alpha=0.15)
    axes[2].set_title('E[c_t]'); axes[2].set_ylabel('E[c_t]')
    axes[2].set_ylim(-0.2, 3.2); axes[2].set_yticks([0, 1, 2, 3]); axes[2].set_yticklabels(LAYER_NAMES)

    acc = (y_true[vm] == pred[vm]).mean() if vm.sum() > 0 else 0
    fig.text(0.5, 0.02, f'Acc: {acc:.4f}', ha='center', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(os.path.join(save_dir, f'{well_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_confidence_profile(well_data, cbe, pss, well_name, save_dir):
    fig, axes = plt.subplots(3, 1, figsize=(15, 14), sharex=True)
    fig.suptitle(f'{well_name} - ASC weight (CBE) vs ATC weight (PSS)', fontsize=16, fontweight='bold')

    tvdss = well_data['tvdss']; y_true = well_data['labels']; vm = well_data['valid_mask']
    n = len(tvdss); x_axis = range(n)

    axes[0].scatter(x_axis, -tvdss, c=y_true, cmap='viridis', s=10, alpha=0.8)
    axes[0].set_title('True Stratigraphy'); axes[0].set_ylabel('TVDSS (inv)')
    cbar = plt.colorbar(axes[0].collections[0], ax=axes[0])
    cbar.set_ticks([0, 1, 2, 3]); cbar.set_ticklabels(LAYER_NAMES)

    axes[1].plot(x_axis, cbe, color='#E74C3C', linewidth=1.2, label='ASC weight (CBE)')
    axes[1].fill_between(x_axis, 0, cbe, color='#E74C3C', alpha=0.15)
    axes[1].set_ylabel('CBE'); axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc='upper right'); axes[1].grid(True, alpha=0.3)

    axes[2].plot(x_axis, pss, color='#2980B9', linewidth=1.2, label='ATC weight (PSS)')
    axes[2].fill_between(x_axis, 0, pss, color='#2980B9', alpha=0.15)
    axes[2].set_ylabel('PSS'); axes[2].set_xlabel('Depth Index')
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc='upper right'); axes[2].grid(True, alpha=0.3)

    corr = np.corrcoef(cbe[vm], pss[vm])[0, 1] if vm.sum() > 1 else 0.0
    fig.text(0.5, 0.01, f'CBE-PSS Pearson r = {corr:.4f}',
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(os.path.join(save_dir, f'{well_name}_confidence.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curves(history_list, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('UGCNet Training Curves (CE + ASC + ATC)', fontsize=16, fontweight='bold')

    for fid, hist in enumerate(history_list):
        if not hist.get('train_loss'):
            continue
        epochs = list(range(1, len(hist['train_loss']) + 1))
        label = f'Fold {fid + 1}'

        axes[0, 0].plot(epochs, hist['train_loss'], linewidth=1, label=f'{label} Total')
        axes[0, 0].plot(epochs, hist['ce_loss'], '--', linewidth=0.8, label=f'{label} CE')
        axes[0, 0].plot(epochs, hist['asc_loss'], ':', linewidth=0.8, label=f'{label} ASC')
        axes[0, 0].plot(epochs, hist['atc_loss'], '-.', linewidth=0.8, label=f'{label} ATC')
        axes[0, 0].set_title('Loss Decomposition'); axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss'); axes[0, 0].legend(fontsize=7)

        if hist.get('val_acc'):
            val_epochs = list(range(VAL_FREQUENCY, VAL_FREQUENCY + VAL_FREQUENCY * len(hist['val_acc']), VAL_FREQUENCY))
            axes[0, 1].plot(val_epochs, hist['val_acc'], 'o-', markersize=3, linewidth=1, label=label)
        axes[0, 1].set_title('Validation Accuracy'); axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy'); axes[0, 1].legend(fontsize=7)

        if hist.get('val_f1'):
            val_epochs = list(range(VAL_FREQUENCY, VAL_FREQUENCY + VAL_FREQUENCY * len(hist['val_f1']), VAL_FREQUENCY))
            axes[1, 0].plot(val_epochs, hist['val_f1'], 's-', markersize=3, linewidth=1, label=label)
        axes[1, 0].set_title('Validation F1'); axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1'); axes[1, 0].legend(fontsize=7)

        axes[1, 1].plot(epochs, hist['learning_rate'], linewidth=1, label=label)
        axes[1, 1].set_title('Learning Rate'); axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR'); axes[1, 1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LAYER_NAMES, yticklabels=LAYER_NAMES, ax=ax)
    ax.set_title('Confusion Matrix (UGCNet paper)')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150); plt.close()


def main():
    print("=" * 80)
    print("UGCNet - Paper-consistent (no post-processing)")
    print("=" * 80)
    print(f"\n  Validation wells: {VAL_WELL_NAMES}")
    print(f"  F={N_FEATURES}, layers={NUM_LAYERS}, gnn={GNN_LAYERS}, "
          f"LR={LR}, lambda_asc={ASC_CONFIG['base_weight']}, lambda_atc={ATC_CONFIG['base_weight']}")
    print("=" * 80)

    FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
    PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')
    CONFIDENCE_DIR = os.path.join(OUTPUT_DIR, 'confidence_profiles')
    for d in (OUTPUT_DIR, FIGURES_DIR, PREDICTIONS_DIR, CONFIDENCE_DIR):
        os.makedirs(d, exist_ok=True)

    print("\n[1/7] Loading 4ch single-resolution data...")
    well_data_list, coords, edge_index, edge_attr, global_min, global_max = load_and_preprocess_data()
    all_well_names = [w['name'] for w in well_data_list]
    print(f"[OK] {len(well_data_list)} wells")

    val_found = [n for n in VAL_WELL_NAMES if n in all_well_names]
    val_not_found = [n for n in VAL_WELL_NAMES if n not in all_well_names]
    if val_not_found:
        print(f"[WARN] Wells not found: {val_not_found}")
    val_indices = [all_well_names.index(n) for n in val_found]
    train_names = [n for n in all_well_names if n not in val_found]
    print(f"  Validation ({len(val_found)}): {val_found}")
    print(f"  Training ({len(train_names)}): {train_names}")

    print("\n[2/7] Initializing UGCNet...")
    model = UGCNet(
        input_channels=N_FEATURES, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, nhead=NHEAD, n_classes=4, n_gnn_layers=GNN_LAYERS,
        dropout=0.1, depth_scale=DEPTH_SCALE,
        asc_config=ASC_CONFIG, atc_config=ATC_CONFIG,
    )
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    print("\n[3/7] Training (single split) ...")
    trainer = Trainer(model, well_data_list, edge_index, edge_attr,
                      use_augmentation=True, val_well_names=val_found)
    trainer.train()

    print("\n[4/7] Loading best checkpoint for inference...")
    log_dirs = sorted([d for d in os.listdir(OUTPUT_DIR) if d.startswith('training_logs_')])
    if log_dirs:
        latest_ckpt = os.path.join(OUTPUT_DIR, log_dirs[-1], 'best_model_fold1.pt')
        if os.path.exists(latest_ckpt):
            ckpt = torch.load(latest_ckpt, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"  Loaded epoch {ckpt['epoch']}, val_f1={ckpt['val_f1']:.4f}")
    model.eval(); model.to(DEVICE)
    with torch.no_grad():
        X_all = [w['curves'] for w in well_data_list]
        depth_all = [w['tvdss'] for w in well_data_list]
        preds_all, expected_list_all = model(X_all, edge_index.to(DEVICE), depth_list=depth_all)
        profiles_all = model.compute_confidence_profiles(X_all, edge_index.to(DEVICE), depth_list=depth_all)

    print("\n[5/7] No post-processing (raw argmax predictions).")

    print("\n[6/7] Saving validation well results...")
    all_y_true, all_y_pred = [], []
    val_stats = []

    for vi in val_indices:
        wd = well_data_list[vi]; pf = preds_all[vi]
        ev = expected_list_all[vi]; prof = profiles_all[vi]; wn = all_well_names[vi]
        vm, yt = wd['valid_mask'], wd['labels']

        df = pd.DataFrame({
            'Depth': wd['depth'], 'TVDSS': wd['tvdss'],
            'True_Layer': yt, 'Pred_Layer': pf,
            'Expected_Ect': ev, 'Valid': vm,
            'ASC_weight_CBE': prof['cbe'],
            'ATC_weight_PSS': prof['pss'],
        })
        acc = (yt[vm] == pf[vm]).mean() if vm.sum() > 0 else 0.0
        info = {'well_name': wn, 'accuracy': acc,
                'cbe_mean': prof['cbe'][vm].mean() if vm.sum() > 0 else 0,
                'pss_mean': prof['pss'][vm].mean() if vm.sum() > 0 else 0}
        val_stats.append(info)

        all_y_true.extend(yt[vm].tolist())
        all_y_pred.extend(pf[vm].tolist())
        df.to_excel(os.path.join(PREDICTIONS_DIR, f'{wn}.xlsx'), index=False)
        print(f"  [VAL] {wn}: Acc={acc:.4f}, CBE={info['cbe_mean']:.4f}, PSS={info['pss_mean']:.4f}")

        plot_single_well(wd, pf, ev, wn, FIGURES_DIR)
        plot_confidence_profile(wd, prof['cbe'], prof['pss'], wn, CONFIDENCE_DIR)

    stats_df = pd.DataFrame(val_stats)
    stats_df.to_excel(os.path.join(OUTPUT_DIR, 'well_accuracy.xlsx'), index=False)
    val_acc_avg = np.mean([s['accuracy'] for s in val_stats]) if val_stats else 0.0
    print(f"\n  Validation avg accuracy: {val_acc_avg:.4f}")

    p, r, f1, s = precision_recall_fscore_support(all_y_true, all_y_pred, labels=[0, 1, 2, 3],
                                                  average=None, zero_division=0)
    metrics_df = pd.DataFrame({'Layer': LAYER_NAMES, 'Precision': p, 'Recall': r,
                               'F1-score': f1, 'Support': s})
    metrics_df.to_excel(os.path.join(OUTPUT_DIR, 'per_class_metrics.xlsx'), index=False)
    print("\n--- Per-Class (Validation) ---")
    print(metrics_df.to_string(index=False))

    overall = (np.array(all_y_true) == np.array(all_y_pred)).mean()
    print(f"\nValidation Accuracy: {overall:.4f} ({overall*100:.2f}%)")

    print("\n[7/7] Generating analysis plots...")
    plot_confusion_matrix(all_y_true, all_y_pred, OUTPUT_DIR)

    log_dirs = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR)
                if d.startswith('training_logs_')]
    if log_dirs:
        latest_dir = max(log_dirs, key=os.path.getmtime)
        fold_histories = []
        for fn in sorted(os.listdir(latest_dir)):
            if fn.endswith('_history.json'):
                with open(os.path.join(latest_dir, fn), 'r') as f:
                    fold_histories.append(json.load(f))
        if fold_histories:
            plot_loss_curves(fold_histories, latest_dir)

    print("\n" + "=" * 80)
    print(f"[DONE] Validation Accuracy: {overall:.4f} ({overall*100:.2f}%)")
    print("=" * 80)
    print(f"\nResults: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
