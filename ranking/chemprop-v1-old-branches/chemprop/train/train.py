import logging
from typing import Callable
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset, AtomBondScaler
from chemprop.models import MoleculeModel
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR
from chemprop.utils import features_to_group_ids

def supervised_nt_xent(z_embed, group_ids, temperature=0.2):
    """
    Supervised InfoNCE (NT-Xent) loss for zeolite groups.
    Works even when some groups have only one sample.
    """
    # Normalize embeddings
    z = F.normalize(z_embed, dim=1)
    sim = torch.mm(z, z.t()) / temperature      # cosine similarity matrix
    labels = (group_ids.unsqueeze(0) == group_ids.unsqueeze(1)).float()
    mask = torch.eye(labels.size(0), device=z.device).bool()
    labels = labels.masked_fill(mask, 0)        # remove self-comparisons

    # For each sample, compute log-prob of its positives among all others
    exp_sim = torch.exp(sim) * (~mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

    # Average only over positive pairs
    loss = -(labels * log_prob).sum(dim=1) / (labels.sum(dim=1) + 1e-9)
    return loss.mean()

def train(
    model: MoleculeModel,
    data_loader: MoleculeDataLoader,
    loss_func: Callable,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    args: TrainArgs,
    n_iter: int = 0,
    atom_bond_scaler: AtomBondScaler = None,
    logger: logging.Logger = None,
    writer: SummaryWriter = None,
) -> int:
    """Trains a model for one epoch (ranking + optional contrastive regularization)."""
    debug = logger.debug if logger is not None else print
    model.train()

    loss_sum, iter_count = 0, 0
    lambda_contrastive = args.lambda_contrastive
    margin = getattr(args, "contrastive_margin", 0.2)

    for batch in tqdm(data_loader, total=len(data_loader), leave=False):
        batch: MoleculeDataset
        mol_batch, features_batch, target_batch, mask_batch, atom_descriptors_batch, atom_features_batch, bond_descriptors_batch, bond_features_batch, constraints_batch, data_weights_batch = \
            batch.batch_graph(), batch.features(), batch.targets(), batch.mask(), batch.atom_descriptors(), \
            batch.atom_features(), batch.bond_descriptors(), batch.bond_features(), batch.constraints(), batch.data_weights()

        # Prepare tensors
        mask_batch = np.transpose(mask_batch).tolist()
        masks = torch.tensor(mask_batch, dtype=torch.bool).to(args.device)
        targets = torch.tensor([[0 if x is None else x for x in tb] for tb in target_batch]).to(args.device)

        # === Forward ===
        model.zero_grad()
        preds = model(
            mol_batch,
            features_batch,
            atom_descriptors_batch,
            atom_features_batch,
            bond_descriptors_batch,
            bond_features_batch,
            constraints_batch,
            None,
        )

        # === Ranking (ListNet) loss ===
        if args.dataset_type == "ranking" and args.loss_function == "listnet":
            logits_row = preds[:, 0] if preds.dim() == 2 else preds.squeeze()
            y_row = targets[:, 0].float() if targets.dim() > 1 else targets.float()
            m_row = masks[:, 0].bool() if masks.dim() > 1 else masks.bool()

            logits_row = logits_row[m_row]
            y_row = y_row[m_row]

            # Zeolite group IDs per sample
            group_ids_full = features_to_group_ids(features_batch=features_batch, device=args.device)
            gids = group_ids_full[m_row]

            uniq, inv = torch.unique(gids, return_inverse=True)
            S = uniq.numel()
            if S == 0:
                continue

            temperature = getattr(args, "listnet_temperature", 0.3)
            slate_losses = []

            # -------- Correct per-slate averaging (true normalization across slates) --------
            for s_id in range(S):
                idx = (inv == s_id)
                if idx.sum() < 2:
                    continue

                s = logits_row[idx] / temperature
                t = y_row[idx]

                ln = loss_func(
                    s.unsqueeze(0), 
                    t.unsqueeze(0),
                    eps=getattr(args, "listnet_eps", 1e-10),
                    padded_value_indicator=getattr(args, "listnet_pad_val", -1.0),
                )

                slate_losses.append(ln.squeeze())

            if not slate_losses:
                continue

            # -------- True normalization across slates --------
            slate_losses = torch.stack(slate_losses)
            ranking_loss = slate_losses.mean()

        if model.zeolite_encoder is not None:
            if isinstance(features_batch, list):
                feat_tensor = torch.from_numpy(np.stack(features_batch)).float().to(args.device)
            else:
                feat_tensor = features_batch.to(args.device).float()

            z_full = model.zeolite_encoder(feat_tensor)  # [N, hidden_size]
        else:
            z_full = None

        # === Contrastive loss using logits as embeddings ===
        # with torch.no_grad():
        #     group_ids_full = features_to_group_ids(features_batch=features_batch, device=args.device)
        # mask = m_row
        # gids_masked = group_ids_full[mask]

        # z_embed = F.normalize(features_batch, dim=1)
        # gids = gids_masked

        # loss_terms = []
        # for gid in gids.unique():
        #     mask_pos = (gids == gid)
        #     mask_neg = ~mask_pos
        #     z_pos = z_embed[mask_pos]
        #     z_neg = z_embed[mask_neg]
        #     ia, ip = torch.randint(0, z_pos.size(0), (2,))
        #     z_anchor, z_positive = z_pos[ia], z_pos[ip]
        #     z_negative = z_neg[torch.randint(0, z_neg.size(0), (1,))]
        #     pos_dist = 1 - F.cosine_similarity(z_anchor, z_positive.unsqueeze(0))
        #     neg_dist = 1 - F.cosine_similarity(z_anchor, z_negative)
        #     loss_terms.append(F.relu(pos_dist - neg_dist + margin))
        
        # unique_gids, counts = gids_masked.unique(return_counts=True)
        # weight_map = (1.0 / counts.float()).to(gids_masked.device)
        # weight_map = weight_map / weight_map.mean()

        # sample_weights = torch.zeros_like(gids_masked, dtype=torch.float)
        # for u, w in zip(unique_gids, weight_map):
        #     sample_weights[gids_masked == u] = w

        # contrastive_loss = supervised_nt_xent(
        #     z_embed=h_masked, 
        #     group_ids=gids_masked, 
        #     temperature=getattr(args, "cl_temperature", 0.2)
        # )
            
        with torch.no_grad():
            group_ids_full = features_to_group_ids(features_batch=features_batch, device=args.device)

        if z_full is not None:
            z_masked = z_full[m_row]
            gids_masked = group_ids_full[m_row]

            z_norm = F.normalize(z_masked, dim=1)

            contrastive_loss = supervised_nt_xent(
                z_embed=z_norm,
                group_ids=gids_masked,
                temperature=getattr(args, "cl_temperature", 0.2),
            )

        #=== Combine losses ===
        total_loss = ranking_loss + lambda_contrastive * contrastive_loss

        loss_sum += total_loss.item()
        iter_count += 1
        total_loss.backward()

        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)

        # Logging
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count if iter_count > 0 else 0
            loss_sum, iter_count = 0, 0
            debug(f"Loss = {loss_avg:.4e}, ListNET = {ranking_loss.item():.4e}, CL = {contrastive_loss.item():.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, LR = {lrs[0]:.4e}")
            if writer is not None:
                writer.add_scalar("train_total", loss_avg, n_iter)
                writer.add_scalar("train_rank", ranking_loss.item(), n_iter)
                writer.add_scalar("param_norm", pnorm, n_iter)
                writer.add_scalar("gradient_norm", gnorm, n_iter)

    return n_iter
