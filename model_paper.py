import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch_geometric.nn import GATv2Conv
from pos_encoding_paper import DepthPositionalEncoding


# =============================================================================
# Causal Transformer encoder  (paper Section 3.3)
# =============================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4   # paper: feed-forward dimension = 512 for d=128
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class CausalTransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers=4, nhead=8, dropout=0.1,
                 depth_scale=1000.0):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.depth_pe = DepthPositionalEncoding(hidden_dim, dropout=dropout, depth_scale=depth_scale)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, nhead, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, depth=None):
        x = self.input_proj(x)
        if depth is not None:
            x = self.depth_pe(x, depth)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# =============================================================================
# Two-layer GAT  (paper Section 3.3: first layer 4 heads, second layer 1 head)
# =============================================================================
class GNNStack(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_dim, hidden_dim // 4, heads=4, concat=True, dropout=0.2))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, concat=True, dropout=0.2))
        self.convs.append(GATv2Conv(hidden_dim, out_dim, heads=1, concat=False, dropout=0.2))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        return x


# =============================================================================
# Uncertainty-aware geological constraints  (paper Sections 3.4 / 3.5)
# =============================================================================
class GeologicalConstraintLoss(nn.Module):
    """Adaptive Sequence Constraint (ASC) + Adaptive Thickness Constraint (ATC).

    ASC adaptive weight uses normalized cumulative binary entropy (CBE).
    ATC adaptive weight uses the prediction stability score (PSS).
    Both weighting signals are detached (stop-gradient).
    """

    def __init__(self, asc_config, atc_config):
        super().__init__()
        self.asc_base_weight = asc_config['base_weight']     # lambda_asc
        self.asc_temperature = asc_config['temperature']     # t
        self.asc_min_conf = asc_config['min_confidence']     # w_min
        self.asc_max_conf = asc_config['max_confidence']     # w_max
        self.asc_exponent = asc_config.get('exponent', 1.0)  # g
        self.asc_margin = asc_config.get('margin', 0.0)

        self.atc_base_weight = atc_config['base_weight']     # lambda_atc
        self.atc_temperature = atc_config['temperature']     # t_atc
        self.atc_min_conf = atc_config['min_confidence']
        self.atc_max_conf = atc_config['max_confidence']
        self.atc_exponent = atc_config.get('exponent', 1.0)
        self.atc_min_length = atc_config.get('min_layer_length', 3)  # delta_min
        self.atc_window = atc_config.get('stability_window', 5)      # W

    # ---- ASC weighting: normalized cumulative binary entropy -> weight ----
    def asc_weight(self, probs):
        """w = clamp( (1 - (H_asc)^t)^g , w_min, w_max ),  detached."""
        probs32 = probs.float()
        eps = 1e-8

        Q1 = probs32[:, :, 1:].sum(dim=-1)   # P(class >= IV)
        Q2 = probs32[:, :, 2:].sum(dim=-1)   # P(class >= V)
        Q3 = probs32[:, :, 3]                # P(class >= VI)

        Q1 = torch.clamp(Q1, eps, 1.0 - eps)
        Q2 = torch.clamp(Q2, eps, 1.0 - eps)
        Q3 = torch.clamp(Q3, eps, 1.0 - eps)

        def _binary_entropy(q):
            return -(q * torch.log(q) + (1.0 - q) * torch.log(1.0 - q))

        avg_entropy = (_binary_entropy(Q1) + _binary_entropy(Q2) + _binary_entropy(Q3)) / 3.0
        norm_entropy = avg_entropy / math.log(2.0)            # H_asc in [0, 1]

        sharpened = norm_entropy ** self.asc_temperature       # H_asc^t
        raw_w = (1.0 - sharpened) ** self.asc_exponent         # (1 - H^t)^g
        w = torch.clamp(raw_w, self.asc_min_conf, self.asc_max_conf)
        return w.detach().type_as(probs)

    # ---- ATC weighting: prediction stability score (PSS) -> weight ----
    def atc_weight(self, probs):
        """w^atc = clamp( s^{t_atc}, w_min, w_max ),  detached."""
        probs32 = probs.float()
        eps = 1e-8
        B, L, C = probs32.shape

        max_probs = probs32.max(dim=-1).values
        window = min(self.atc_window, L)
        if L <= 2:
            raw_s = torch.ones_like(max_probs)
        else:
            half = window // 2
            mp = max_probs.unsqueeze(1)
            mp_padded = F.pad(mp, (half, half), mode='reflect')
            kernel = torch.ones(1, 1, window, device=probs32.device) / window
            local_mean = F.conv1d(mp_padded, kernel).squeeze(1)
            raw_s = 1.0 - torch.abs(max_probs - local_mean)    # PSS

        raw_s = torch.clamp(raw_s, eps, 1.0)
        sharpened = raw_s ** self.atc_temperature
        if self.atc_exponent != 1.0:
            sharpened = sharpened ** self.atc_exponent
        w = torch.clamp(sharpened, self.atc_min_conf, self.atc_max_conf)
        return w.detach().type_as(probs)

    # ---- ASC loss (Eq. local penalty + total) ----
    def asc_loss(self, probs):
        probs32 = probs.float()
        Q1 = probs32[:, :, 1:4].sum(dim=2, keepdim=True)
        Q2 = probs32[:, :, 2:4].sum(dim=2, keepdim=True)
        Q3 = probs32[:, :, 3:4]
        Q = torch.cat([Q1, Q2, Q3], dim=2)

        delta = Q[:, :-1, :] - Q[:, 1:, :]               # Q_j - Q_{j+1}
        hinge = F.relu(delta - self.asc_margin)          # max(0, .)

        w = self.asc_weight(probs)
        pair_weight = (w[:, :-1] + w[:, 1:]) / 2.0
        pair_weight = pair_weight.unsqueeze(-1).expand_as(hinge)

        weighted_hinge = hinge * pair_weight
        # mean over (pairs x 3 thresholds) == 1/(3N); scaled by lambda_asc
        return (weighted_hinge.mean() * self.asc_base_weight).type_as(probs)

    # ---- ATC loss: differentiable soft-persistence penalty ----
    #   l_{i,j}^thick = w^atc_{i,j} * (1 - pi_{i,j})
    #   pi_{i,j} = mean over N_delta(j)={+-1,...,+-(delta_min-1)} of <p_j, p_{j+t}>
    # No argmax / hard segments -> fully differentiable w.r.t. probs.
    def atc_loss(self, probs):
        probs32 = probs.float()                       # (B, L, C)
        B, L, C = probs32.shape
        w = self.atc_weight(probs)                    # (B, L) detached PSS weight

        radius = self.atc_min_length - 1              # N_delta radius (delta_min - 1)
        if radius < 1 or L < 2:
            return torch.tensor(0.0, device=probs.device, dtype=probs.dtype)

        sim_sum = torch.zeros(B, L, device=probs32.device)   # sum_t <p_j, p_{j+t}>
        cnt = torch.zeros(B, L, device=probs32.device)       # |valid neighbors|
        for t in range(1, radius + 1):
            dot = (probs32[:, :L - t, :] * probs32[:, t:, :]).sum(-1)   # (B, L-t): pair (j, j+t)
            ones = torch.ones_like(dot)
            sim_sum = sim_sum + F.pad(dot, (0, t)) + F.pad(dot, (t, 0))
            cnt = cnt + F.pad(ones, (0, t)) + F.pad(ones, (t, 0))

        pi = sim_sum / torch.clamp(cnt, min=1.0)      # local persistence in [0, 1]
        penalty = w * (1.0 - pi)                       # point-wise thickness penalty
        loss = penalty.mean()                          # (1/N) sum_{i,j}
        return (loss * self.atc_base_weight).type_as(probs)   # lambda_atc

    def forward(self, probs):
        return self.asc_loss(probs), self.atc_loss(probs)


# =============================================================================
# UGCNet
# =============================================================================
class UGCNet(nn.Module):
    """UGCNet (paper-consistent).

    Transformer-GAT backbone + standard cross-entropy + ASC + ATC.
    No prediction post-processing (no mode filter, no monotonic correction).
    """

    def __init__(self, input_channels=4, hidden_dim=128, num_layers=4, nhead=8,
                 n_classes=4, n_gnn_layers=2, dropout=0.1, depth_scale=1000.0,
                 asc_config=None, atc_config=None):
        super().__init__()

        if asc_config is None:
            asc_config = {'base_weight': 10.0, 'temperature': 1.5,
                          'min_confidence': 0.15, 'max_confidence': 1.0,
                          'exponent': 1.2, 'margin': 0.0}
        if atc_config is None:
            atc_config = {'base_weight': 0.1, 'temperature': 1.0,
                          'min_confidence': 0.1, 'max_confidence': 1.0,
                          'exponent': 1.0, 'min_layer_length': 3,
                          'stability_window': 5}

        self.constraints = GeologicalConstraintLoss(asc_config, atc_config)

        self.transformer = CausalTransformerEncoder(
            input_size=input_channels, hidden_dim=hidden_dim,
            num_layers=num_layers, nhead=nhead, dropout=dropout, depth_scale=depth_scale,
        )
        self.gnn = GNNStack(hidden_dim, hidden_dim, hidden_dim, num_layers=n_gnn_layers)

        # Classifier input = [h_{i,j} ; w_tilde_i]  (paper Eq. for p_{i,j}).
        # Two-layer MLP with ReLU + dropout. No separate depth-MLP branch:
        # depth information is supplied solely through TVDSS positional encoding.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def _encode(self, x_list, edge_index, depth_list=None):
        device = next(self.parameters()).device
        B = len(x_list)
        max_len = max(x.shape[0] for x in x_list)
        n_features = x_list[0].shape[1]

        x_padded = torch.zeros(B, max_len, n_features, device=device)
        depth_padded = torch.zeros(B, max_len, device=device) if depth_list is not None else None
        for i in range(B):
            x_arr = x_list[i]
            if not isinstance(x_arr, torch.Tensor):
                x_arr = torch.tensor(x_arr, dtype=torch.float, device=device)
            n = x_arr.shape[0]
            x_padded[i, :n] = x_arr
            if depth_padded is not None:
                d_arr = depth_list[i]
                if not isinstance(d_arr, torch.Tensor):
                    d_arr = torch.tensor(d_arr, dtype=torch.float, device=device)
                depth_padded[i, :n] = d_arr[:n]

        tf_out = self.transformer(x_padded, depth=depth_padded)

        per_well_feats = []
        well_embeddings = []
        for i in range(B):
            seq_len = x_list[i].shape[0]
            feat = tf_out[i, :seq_len]
            per_well_feats.append(feat)
            well_embeddings.append(feat.mean(dim=0))   # depth-wise average pooling

        well_embeddings = torch.stack(well_embeddings, dim=0)
        well_embeddings_agg = self.gnn(well_embeddings, edge_index)

        emissions_list = []
        for i in range(B):
            seq_len = per_well_feats[i].shape[0]
            gnn_feat = well_embeddings_agg[i].unsqueeze(0).repeat(seq_len, 1)
            combined = torch.cat([per_well_feats[i], gnn_feat], dim=-1)
            emissions_list.append(self.classifier(combined))

        return emissions_list

    def forward(self, x_list, edge_index, labels_list=None, valid_mask_list=None,
                depth_list=None, return_components=False):
        device = next(self.parameters()).device
        B = len(x_list)
        emissions_list = self._encode(x_list, edge_index, depth_list=depth_list)

        if labels_list is not None:
            total_loss = 0.0
            ce_sum = 0.0
            asc_sum = 0.0
            atc_sum = 0.0

            for i in range(B):
                seq_len = len(labels_list[i])
                emission = emissions_list[i]
                label = torch.tensor(labels_list[i], dtype=torch.long, device=device)

                if valid_mask_list is not None and i < len(valid_mask_list):
                    valid_mask = torch.tensor(valid_mask_list[i][:seq_len], dtype=torch.bool, device=device)
                    valid_emission = emission[valid_mask]
                    valid_label = label[valid_mask]
                    if len(valid_label) == 0:
                        continue
                else:
                    valid_emission = emission
                    valid_label = label

                # Standard cross-entropy (paper L_ce).
                ce_loss = F.cross_entropy(valid_emission, valid_label)

                probs = F.softmax(valid_emission, dim=-1)
                asc_loss, atc_loss = self.constraints(probs.unsqueeze(0))

                ce_sum += ce_loss
                asc_sum += asc_loss
                atc_sum += atc_loss
                total_loss += ce_loss + asc_loss + atc_loss

            n = B if B > 0 else 1
            if return_components:
                return total_loss / n, {
                    'ce': (ce_sum / n).item() if torch.is_tensor(ce_sum) else 0.0,
                    'asc': (asc_sum / n).item() if torch.is_tensor(asc_sum) else 0.0,
                    'atc': (atc_sum / n).item() if torch.is_tensor(atc_sum) else 0.0,
                }
            return total_loss / n if B > 0 else torch.tensor(0.0, device=device)

        # Inference: raw argmax. No post-processing.
        pred_list = []
        expected_list = []
        class_values = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device)
        for emission in emissions_list:
            probs = F.softmax(emission.detach(), dim=-1)
            expected = (probs * class_values).sum(dim=-1).cpu().numpy()
            expected_list.append(expected)
            pred_list.append(torch.argmax(emission, dim=-1).cpu().numpy())

        return pred_list, expected_list

    def compute_confidence_profiles(self, x_list, edge_index, depth_list=None):
        """Per-well ASC weight (CBE-based) and ATC weight (PSS-based) profiles.

        Used only for visualization/analysis, not for modifying predictions.
        """
        self.eval()
        emissions_list = self._encode(x_list, edge_index, depth_list=depth_list)
        profiles = []
        with torch.no_grad():
            for emission in emissions_list:
                probs = F.softmax(emission, dim=-1).unsqueeze(0)
                cbe = self.constraints.asc_weight(probs).squeeze(0)
                pss = self.constraints.atc_weight(probs).squeeze(0)
                pred = torch.argmax(emission, dim=-1)
                profiles.append({
                    'cbe': cbe.cpu().numpy(),
                    'pss': pss.cpu().numpy(),
                    'pred': pred.cpu().numpy(),
                    'probs': probs.squeeze(0).cpu().numpy(),
                })
        return profiles
