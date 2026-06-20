import torch
import numpy as np
import json
import os
import random
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial.distance import cdist
from config_paper import *


def augment_curves(curves, noise_std=AUG_NOISE_STD):
    noise = np.random.normal(0, noise_std, curves.shape).astype(np.float32)
    augmented = curves + noise
    return np.clip(augmented, 0, 1)


class Trainer:
    """Paper-consistent trainer: AdamW + warmup + cosine restarts + early stopping.

    No SWA, no prediction post-processing. Predictions are the model's raw argmax.
    """

    def __init__(self, model, well_data_list, edge_index, edge_attr,
                 use_augmentation=True, val_well_names=None):
        self.model = model
        self.edge_index = edge_index.to(DEVICE)
        self.edge_attr = edge_attr.to(DEVICE) if edge_attr is not None else None
        self.device = DEVICE
        self.model.to(self.device)
        self.use_augmentation = use_augmentation
        self.well_data_list = well_data_list

        self.X_list = [w['curves'] for w in well_data_list]
        self.y_list = [w['labels'] for w in well_data_list]
        self.valid_mask_list = [w.get('valid_mask', np.ones_like(w['labels'], dtype=bool)) for w in well_data_list]
        self.well_names = [w['name'] for w in well_data_list]
        self.depth_list = [w['tvdss'] for w in well_data_list]

        self.coords = np.array([[w['X'], w['Y']] for w in well_data_list])
        self.altitudes = np.array([w['kb'] for w in well_data_list]).reshape(-1, 1)

        self.val_well_names = val_well_names
        if val_well_names is not None:
            self.val_idx = [i for i, name in enumerate(self.well_names) if name in val_well_names]
            self.train_idx = [i for i in range(len(self.well_names)) if i not in self.val_idx]
        else:
            self.val_idx = None
            self.train_idx = None

        self.history = {'fold_metrics': []}

    # ---- graph construction (paper thresholds) ----
    def _build_spatial_edges(self, indices):
        n_nodes = len(indices)
        if n_nodes == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), None
        subset_coords = self.coords[indices]
        subset_altitudes = self.altitudes[indices]
        plane_dist_matrix = cdist(subset_coords, subset_coords, metric='euclidean')
        altitude_diff_matrix = cdist(subset_altitudes, subset_altitudes, metric='chebyshev')
        edge_list, attr_list = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                if altitude_diff_matrix[i, j] > ALTITUDE_DIFF_THRESH:
                    continue
                if plane_dist_matrix[i, j] > PLANE_DIST_THRESH:
                    continue
                edge_list.append([i, j])
                attr_list.append(1.0 / (plane_dist_matrix[i, j] + 1e-6))
        if len(edge_list) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), None
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t().contiguous()
        edge_attr = torch.tensor(attr_list, dtype=torch.float, device=self.device)
        return edge_index, edge_attr

    def _create_val_subgraph(self, train_indices, val_indices):
        all_indices = train_indices + val_indices
        full_edge_index, full_edge_attr = self._build_spatial_edges(all_indices)
        if full_edge_index.size(1) == 0:
            return full_edge_index, None
        global_to_local = {gi: li for li, gi in enumerate(all_indices)}
        train_local_indices = set(global_to_local[idx] for idx in train_indices)
        valid_edge_indices = []
        for i in range(full_edge_index.size(1)):
            src_local = full_edge_index[0, i].item()
            if src_local in train_local_indices:
                valid_edge_indices.append(i)
        if len(valid_edge_indices) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), None
        filtered_edge_index = full_edge_index[:, valid_edge_indices]
        return filtered_edge_index.to(self.device), None

    def validate(self, train_idx, val_idx):
        self.model.eval()
        all_indices = train_idx + val_idx
        all_X = [self.X_list[i] for i in all_indices]
        all_depth = [self.depth_list[i] for i in all_indices]
        edge_full = self._create_val_subgraph(train_idx, val_idx)[0]
        with torch.no_grad():
            preds, _ = self.model(all_X, edge_full, depth_list=all_depth)
        val_local = [all_indices.index(v) for v in val_idx]
        y_true_flat, y_pred_flat = [], []
        for li, vid in zip(val_local, val_idx):
            vm = self.valid_mask_list[vid]
            yt = self.y_list[vid]
            ps = preds[li]
            y_true_flat.extend(yt[vm].tolist())
            y_pred_flat.extend([ps[j] for j in range(len(vm)) if vm[j]])
        if len(y_true_flat) == 0:
            return 0.0, 0.0
        return (accuracy_score(y_true_flat, y_pred_flat),
                f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0))

    def _save_config_snapshot(self, result_dir):
        config_dict = {
            'hidden_dim': HIDDEN_DIM, 'nhead': NHEAD, 'num_layers': NUM_LAYERS,
            'gnn_layers': GNN_LAYERS, 'depth_scale': DEPTH_SCALE,
            'n_features': N_FEATURES,
            'epochs': EPOCHS, 'lr': LR,
            'warmup_epochs': WARMUP_EPOCHS, 't_0_restart': T_0_RESTART,
            't_mult_factor': T_MULT_FACTOR, 'lr_min': LR_MIN,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'val_frequency': VAL_FREQUENCY, 'aug_noise_std': AUG_NOISE_STD,
            'weight_decay': WEIGHT_DECAY,
            'loss': 'standard cross-entropy + ASC + ATC (no focal, no class weights)',
            'asc_config': ASC_CONFIG,
            'atc_config': ATC_CONFIG,
            'val_well_names': self.val_well_names,
        }
        with open(os.path.join(result_dir, 'config_snapshot.json'), 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

    def train(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(OUTPUT_DIR, f'training_logs_{timestamp}')
        os.makedirs(result_dir, exist_ok=True)
        self._save_config_snapshot(result_dir)

        if self.val_idx is not None:
            return self._train_single_split(result_dir)

        max_possible_splits = max(2, min(K_FOLDS, len(self.X_list) // 2))
        fold_results = []
        all_preds = [np.zeros_like(y) for y in self.y_list]

        from sklearn.model_selection import StratifiedKFold, KFold
        dominant_layers = []
        for i, y in enumerate(self.y_list):
            valid_mask = self.valid_mask_list[i]
            dominant = np.bincount(y[valid_mask]).argmax() if valid_mask.sum() > 0 else 0
            dominant_layers.append(dominant)
        min_class_count = min(np.bincount(dominant_layers))

        if min_class_count >= max_possible_splits:
            kf = StratifiedKFold(n_splits=max_possible_splits, shuffle=True, random_state=42)
            fold_generator = kf.split(self.X_list, dominant_layers)
        else:
            kf = KFold(n_splits=max_possible_splits, shuffle=True, random_state=42)
            fold_generator = kf.split(self.X_list)

        for fold, (train_idx, val_idx) in enumerate(fold_generator):
            train_idx, val_idx = train_idx.tolist(), val_idx.tolist()
            self._run_fold(fold, train_idx, val_idx, result_dir, all_preds)
            fold_results.append(self.current_fold_metrics)

        self.history['fold_metrics'] = fold_results
        avg_f1 = np.mean([r['best_val_f1'] for r in fold_results])
        print(f"\n{'=' * 60}")
        print(f"{max_possible_splits}-fold CV complete: avg best F1 = {avg_f1:.4f}")
        print(f"{'=' * 60}")

        with open(os.path.join(result_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
        return all_preds, [w['name'] for w in self.well_data_list]

    def _train_single_split(self, result_dir):
        train_idx = self.train_idx
        val_idx = self.val_idx
        print(f"\n{'=' * 60}")
        print("Single train/val split (standard CE + ASC + ATC)")
        print(f"{'=' * 60}")
        print(f"Train wells ({len(train_idx)}): {[self.well_names[i] for i in train_idx]}")
        print(f"Val wells ({len(val_idx)}): {[self.well_names[i] for i in val_idx]}")

        all_preds = [np.zeros_like(y) for y in self.y_list]
        self._run_fold(0, train_idx, val_idx, result_dir, all_preds)
        self.history['fold_metrics'] = [self.current_fold_metrics]
        with open(os.path.join(result_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
        return all_preds, [w['name'] for w in self.well_data_list]

    def _model_has_nan(self):
        for _, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                return True
        return False

    def _run_fold(self, fold, train_idx, val_idx, result_dir, all_preds):
        self.model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.99))

        if WARMUP_EPOCHS > 0:
            def warmup_lambda(epoch):
                if epoch < WARMUP_EPOCHS:
                    return (epoch + 1) / WARMUP_EPOCHS
                return 1.0
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
        else:
            warmup_scheduler = None

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0_RESTART, T_mult=T_MULT_FACTOR, eta_min=LR_MIN)

        scaler = torch.amp.GradScaler('cuda')

        best_val_f1 = 0.0
        best_epoch = 0
        patience_counter = 0
        clean_checkpoint_path = None
        nan_since_epoch = -1

        fold_history = {
            'train_loss': [], 'ce_loss': [], 'asc_loss': [], 'atc_loss': [],
            'val_acc': [], 'val_f1': [], 'learning_rate': [],
        }

        for epoch in range(EPOCHS):
            self.model.train()
            optimizer.zero_grad()

            shuffled = train_idx.copy()
            random.shuffle(shuffled)
            n_mb = max(1, (len(shuffled) + WELLS_PER_MB - 1) // WELLS_PER_MB)
            total_loss_val = 0.0
            ce_accum = asc_accum = atc_accum = 0.0
            valid_mb = 0
            nan_mb = 0

            for mb in range(n_mb):
                mb_wells = shuffled[mb * WELLS_PER_MB:(mb + 1) * WELLS_PER_MB]
                mb_X = [self.X_list[i] for i in mb_wells]
                mb_y = [self.y_list[i] for i in mb_wells]
                mb_vm = [self.valid_mask_list[i] for i in mb_wells]
                mb_depth = [self.depth_list[i] for i in mb_wells]

                if self.use_augmentation:
                    mb_X = [augment_curves(x) for x in mb_X]

                mb_edge_index, _ = self._build_spatial_edges(mb_wells)

                with torch.amp.autocast('cuda'):
                    loss, components = self.model(
                        mb_X, mb_edge_index, mb_y, mb_vm, depth_list=mb_depth,
                        return_components=True)

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_mb += 1
                    del mb_X, mb_y, mb_vm, mb_depth, mb_edge_index, loss
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                total_loss_val += loss.item()
                ce_accum += components['ce']
                asc_accum += components['asc']
                atc_accum += components['atc']
                valid_mb += 1
                del mb_X, mb_y, mb_vm, mb_depth, mb_edge_index, loss

            if valid_mb == 0:
                if nan_since_epoch < 0:
                    nan_since_epoch = epoch + 1
                if epoch - nan_since_epoch + 1 > 10:
                    print("  [FATAL] All NaN for 10 consecutive epochs, stopping")
                    break
                print(f"  [WARN] Epoch {epoch+1}: all micro-batches NaN, skipping update")
                for key in ('train_loss', 'ce_loss', 'asc_loss', 'atc_loss'):
                    fold_history[key].append(float('nan'))
                fold_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
                if warmup_scheduler is not None and epoch < WARMUP_EPOCHS:
                    warmup_scheduler.step()
                else:
                    scheduler.step(epoch - WARMUP_EPOCHS if epoch >= WARMUP_EPOCHS else epoch)
                continue

            nan_since_epoch = -1

            n = valid_mb
            avg_loss = total_loss_val / n
            avg_ce = ce_accum / n
            avg_asc = asc_accum / n
            avg_atc = atc_accum / n
            current_lr = optimizer.param_groups[0]['lr']

            if warmup_scheduler is not None and epoch < WARMUP_EPOCHS:
                warmup_scheduler.step()
            else:
                scheduler.step(epoch - WARMUP_EPOCHS if epoch >= WARMUP_EPOCHS else epoch)

            fold_history['train_loss'].append(avg_loss)
            fold_history['ce_loss'].append(avg_ce)
            fold_history['asc_loss'].append(avg_asc)
            fold_history['atc_loss'].append(avg_atc)
            fold_history['learning_rate'].append(current_lr)

            if (epoch + 1) % VAL_FREQUENCY == 0:
                val_acc, val_f1 = self.validate(train_idx, val_idx)
                fold_history['val_acc'].append(val_acc)
                fold_history['val_f1'].append(val_f1)

                tag = f", NaN:{nan_mb}" if nan_mb > 0 else ""
                print(f"Epoch {epoch + 1:3d}/{EPOCHS}, Total: {avg_loss:.4f} "
                      f"(CE:{avg_ce:.4f} ASC:{avg_asc:.4f} ATC:{avg_atc:.4f}), "
                      f"Acc: {val_acc:.4f}, F1: {val_f1:.4f}, LR: {current_lr:.6f}{tag}")

                if val_f1 > best_val_f1:
                    if self._model_has_nan():
                        print("    [SKIP] Model has NaN params, not saving checkpoint")
                    else:
                        if clean_checkpoint_path is not None:
                            try:
                                os.remove(clean_checkpoint_path)
                            except OSError:
                                pass
                        best_val_f1 = val_f1
                        best_epoch = epoch + 1
                        patience_counter = 0

                        all_indices = train_idx + val_idx
                        all_X = [self.X_list[i] for i in all_indices]
                        all_depth = [self.depth_list[i] for i in all_indices]
                        val_edge = self._create_val_subgraph(train_idx, val_idx)[0]
                        val_local = [all_indices.index(v) for v in val_idx]
                        with torch.no_grad():
                            preds, _ = self.model(all_X, val_edge, depth_list=all_depth)
                        for li, vid in zip(val_local, val_idx):
                            sl = len(self.y_list[vid])
                            all_preds[vid][:sl] = preds[li][:sl]

                        clean_checkpoint_path = os.path.join(result_dir, f'best_model_fold{fold+1}.pt')
                        torch.save({
                            'epoch': best_epoch,
                            'model_state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                            'val_f1': best_val_f1,
                        }, clean_checkpoint_path)
                else:
                    patience_counter += 1

                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        with open(os.path.join(result_dir, f'fold{fold+1}_history.json'), 'w') as f:
            json.dump(fold_history, f, indent=2)

        self.current_fold_metrics = {
            'fold': fold + 1, 'best_val_f1': best_val_f1, 'best_epoch': best_epoch,
        }
        print(f"  Fold {fold + 1} best F1: {best_val_f1:.4f} at epoch {best_epoch}")
