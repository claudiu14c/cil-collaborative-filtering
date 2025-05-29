#!/usr/bin/env python3

"""
scdpp_hybrid.py

Train the full hybrid model (baseline + SVD++ + neighborhood) and log per-epoch RMSE to CSV.

Usage example:
$ python hybrid_submit.py \
    --factors 20 \
    --lr1 0.007 \
    --lr2 0.007 \
    --lr3 0.001 \
    --reg1 0.005 \
    --reg2 0.015 \
    --reg3 0.015 \
    --k 300 \
    --shrink 100.0 \
    --epochs 30 \
    --seed 42 \
    --output output/hybrid_grid_results.csv
"""

import os
import argparse
import numpy as np
from collections import defaultdict
from sklearn.metrics import root_mean_squared_error
from helper_functions import read_data_df


def hybrid_pred(model, sids, pids, min_rating=1.0, max_rating=5.0):
    """
    Prediction for hybrid model (Eq. 16).
    """
    mu = model['mu']
    b_u, b_i = model['b_u'], model['b_i']
    p, q, y = model['p'], model['q'], model['y']
    w, c = model['w'], model['c']
    user2ind = model['user2ind']
    item2ind = model['item2ind']
    implicit = model['implicit']
    neighbors = model['neighbors']
    num_factors = model['num_factors']

    preds = []
    for sid, pid in zip(sids, pids):
        if sid in user2ind and pid in item2ind:
            u = user2ind[sid]; i = item2ind[pid]
            # SVD++ term
            Nu = implicit[u]
            sqrt_Nu = np.sqrt(len(Nu)) if Nu else 1.0
            imp = np.sum(y[Nu], axis=0) / sqrt_Nu if Nu else np.zeros(num_factors)
            svdpp_term = q[i].dot(p[u] + imp)
            # neighborhood
            Rk = [j for j in neighbors[i] if j in implicit[u]]
            Nk = Rk  # since N(u)=R(u)
            sqrt_Rk = np.sqrt(len(Rk)) if Rk else 1.0
            sqrt_Nk = np.sqrt(len(Nk)) if Nk else 1.0
            neigh_explicit = sum((model['implicit_ratings'][u][j] - (mu + b_u[u] + b_i[j])) * w[i][idx]
                                 for idx, j in enumerate(Rk)) / sqrt_Rk
            neigh_implicit = sum(c[i][idx] for idx, j in enumerate(Nk)) / sqrt_Nk

            pred = mu + b_u[u] + b_i[i] + svdpp_term + neigh_explicit + neigh_implicit
        else:
            pred = mu + 0 + 0
        # clip
        preds.append(np.clip(pred, min_rating, max_rating))

    return np.array(preds, dtype=np.float32)


def compute_baseline(user_arr, item_arr, rating_arr, n_users, n_items, reg=20.0, n_epochs=10, lr=0.005):
    """
    Compute global mean mu, user biases bu, and item biases bi.
    Solves min_bu,bi sum((r_ui - mu - bu_u - bi_i)^2) + reg*(bu^2 + bi^2).
    """
    mu = np.mean(rating_arr, dtype=np.float32)
    bu = np.zeros(n_users, np.float32)
    bi = np.zeros(n_items, np.float32)
    for _ in range(n_epochs):
        for u, i, r in zip(user_arr, item_arr, rating_arr):
            pred = mu + bu[u] + bi[i]
            err = r - pred
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])
    return mu, bu, bi


def compute_item_neighbors(user_arr, item_arr, rating_arr, mu, bu, bi, n_items, k=300, shrink=100.0):
    """
    Compute top-k neighbors for each item using shrunk Pearson correlation on residuals.
    Returns: dict i -> list of neighbor item indices (length k or fewer).
    """
    # Build item->(user->residual) map
    item_users = defaultdict(dict)
    for u, i, r in zip(user_arr, item_arr, rating_arr):
        resid = r - (mu + bu[u] + bi[i])
        item_users[i][u] = resid

    neighbors = {}
    for i in range(n_items):
        sims = []
        users_i = item_users[i]
        keys_i = set(users_i.keys())
        mean_i = 0.0  # zero-mean residuals
        var_i = sum(v*v for v in users_i.values())
        for j in range(n_items):
            if j == i:
                continue
            users_j = item_users[j]
            common = keys_i & set(users_j.keys())
            n_common = len(common)
            if n_common < 2:
                continue
            # compute covariance & variances
            cov = sum(users_i[u] * users_j[u] for u in common)
            var_j = sum(users_j[u]*users_j[u] for u in common)
            denom = np.sqrt(var_i * var_j)
            if denom <= 0:
                continue
            pearson = cov / denom
            shrunk = (n_common / (n_common + shrink)) * pearson
            sims.append((shrunk, j))
        # pick top-k
        sims.sort(reverse=True, key=lambda x: x[0])
        neighbors[i] = [j for _, j in sims[:k]]
    return neighbors

def train_hybrid(train_df, valid_df=None,
                 num_factors=20,
                 lr1=0.007, lr2=0.007, lr3=0.001,
                 reg1=0.005, reg2=0.015, reg3=0.015,
                 k=300, shrink=100.0,
                 n_epochs=30, seed=42):
    """
    Train hybrid model and log RMSE each epoch to:
      output/hybrid_curve_f{f}_lr{lr1}-{lr2}-{lr3}_reg{reg1}-{reg2}-{reg3}.csv
    """
    np.random.seed(seed)
    os.makedirs('output', exist_ok=True)

    log_file = (f"output/hybrid_curve_f{num_factors}"
                f"_lr{lr1}-{lr2}-{lr3}"
                f"_reg{reg1}-{reg2}-{reg3}.csv")
    with open(log_file, 'w') as f:
        f.write("epoch,rmse\n")

    # remap IDs
    sids = train_df['sid'].unique()
    pids = train_df['pid'].unique()
    user2ind = {sid: u for u, sid in enumerate(sids)}
    item2ind = {pid: i for i, pid in enumerate(pids)}
    n_users, n_items = len(sids), len(pids)

    # index arrays
    user_arr   = train_df['sid'].map(user2ind).values.astype(np.int32)
    item_arr   = train_df['pid'].map(item2ind).values.astype(np.int32)
    rating_arr = train_df['rating'].values.astype(np.float32)

    # 1. Baseline
    mu, bu, bi = compute_baseline(user_arr, item_arr, rating_arr,
                                   n_users, n_items,
                                   reg=reg1, n_epochs=10, lr=lr1)

    # 2. Implicit feedback
    implicit = {u: [] for u in range(n_users)}
    for u, i in zip(user_arr, item_arr):
        implicit[u].append(i)
    Nu_list = [np.array(implicit[u], dtype=np.int32) for u in range(n_users)]
    Nu_count = np.array([len(l) for l in Nu_list], dtype=np.int32)
    sqrt_Nu = np.where(Nu_count>0, np.sqrt(Nu_count, dtype=np.float32), 1.0)

    # 3. Neighborhood structure
    neighbors = compute_item_neighbors(user_arr, item_arr, rating_arr,
                                       mu=mu, bu=bu, bi=bi,
                                       n_items=n_items,
                                       k=k, shrink=shrink)

    # 4. Initialize parameters
    p = np.random.normal(0, 0.1, (n_users, num_factors)).astype(np.float32)
    q = np.random.normal(0, 0.1, (n_items, num_factors)).astype(np.float32)
    y = np.random.normal(0, 0.1, (n_items, num_factors)).astype(np.float32)
    # w and c: dict i -> array(len(neighbors[i]))
    w = {i: np.zeros(len(neighbors[i]), np.float32) for i in range(n_items)}
    c = {i: np.zeros(len(neighbors[i]), np.float32) for i in range(n_items)}

    # 5. Precompute y_sum per user
    y_sum = np.zeros((n_users, num_factors), np.float32)
    for u in range(n_users):
        if Nu_count[u] > 0:
            y_sum[u] = y[Nu_list[u]].sum(axis=0) / sqrt_Nu[u]

    # 6. Build user->item->rating map for residual lookups
    ratings_by_user = [dict() for _ in range(n_users)]
    for u, i, r in zip(user_arr, item_arr, rating_arr):
        ratings_by_user[u][i] = r

    # 7. SGD loop
    n_ratings = rating_arr.shape[0]
    for epoch in range(1, n_epochs+1):
        perm = np.random.permutation(n_ratings)
        for idx in perm:
            u, i, r = user_arr[idx], item_arr[idx], rating_arr[idx]
            # SVD++ component
            imp = y_sum[u]
            svdpp_term = q[i].dot(p[u] + imp)
            # Neighborhood terms
            Rk = [j for j in neighbors[i] if j in ratings_by_user[u]]
            Nk = [j for j in neighbors[i] if j in implicit[u]]
            sqrt_Rk = np.sqrt(len(Rk), dtype=np.float32) if Rk else 1.0
            sqrt_Nk = np.sqrt(len(Nk), dtype=np.float32) if Nk else 1.0
            neigh_explicit = 0.0
            for idx_j, j in enumerate(Rk):
                neigh_explicit += (ratings_by_user[u][j] - (mu + bu[u] + bi[j])) * w[i][idx_j]
            neigh_explicit /= sqrt_Rk
            neigh_implicit = 0.0
            for idx_j, j in enumerate(Nk):
                neigh_implicit += c[i][idx_j]
            neigh_implicit /= sqrt_Nk

            pred = mu + bu[u] + bi[i] + svdpp_term + neigh_explicit + neigh_implicit
            err  = r - pred

            # update baseline terms
            bu[u] += lr1 * (err - reg1 * bu[u])
            bi[i] += lr1 * (err - reg1 * bi[i])
            
            # update SVD++ factors
            p_u_old = p[u].copy()
            p[u] += lr2 * (err * q[i]   - reg2 * p[u])
            q[i] += lr2 * (err * (p_u_old + imp) - reg2 * q[i])
            if Nu_count[u] > 0:
                coeff = lr2 * err / sqrt_Nu[u]
                for j in Nu_list[u]:
                    y[j] += coeff * q[i] - lr2 * reg2 * y[j]
                # update y_sum[u]
                y_sum[u] = y[Nu_list[u]].sum(axis=0) / sqrt_Nu[u]

            # update neighborhood weights
            for idx_j, j in enumerate(Rk):
                basel_res = ratings_by_user[u][j] - (mu + bu[u] + bi[j])
                w[i][idx_j] += lr3 * (err / sqrt_Rk * basel_res - reg3 * w[i][idx_j])
            for idx_j, j in enumerate(Nk):
                c[i][idx_j] += lr3 * (err / sqrt_Nk - reg3 * c[i][idx_j])

        # end of epoch
        print(f"Epoch {epoch}/{n_epochs} completed.", flush=True)

        # evaluate on valid_df:
        if valid_df is not None:
            model = { 
              'mu': mu, 'b_u': bu, 'b_i': bi,
              'p': p, 'q': q, 'y': y,
              'w': w, 'c': c,
              'user2ind': user2ind, 'item2ind': item2ind,
              'implicit': implicit,
              'implicit_ratings': ratings_by_user,
              'neighbors': neighbors,
              'num_factors': num_factors
            }
            preds = hybrid_pred(model,
                                valid_df['sid'].values,
                                valid_df['pid'].values)
            rmse = root_mean_squared_error(valid_df['rating'].values, preds)
            print(f"Validation RMSE after {epoch} epochs: {rmse:.4f}", flush=True)
            with open(log_file, 'a') as f:
                f.write(f"{epoch},{rmse}\n")

    # return final model
    return {
        'mu': mu, 'b_u': bu, 'b_i': bi,
        'p': p, 'q': q, 'y': y,
        'w': w, 'c': c,
        'user2ind': user2ind, 'item2ind': item2ind,
        'implicit': implicit,
        'implicit_ratings': ratings_by_user,
        'neighbors': neighbors,
        'num_factors': num_factors
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--factors', type=int, default=20)
    p.add_argument('--lr1', type=float, default=0.007)
    p.add_argument('--lr2', type=float, default=0.007)
    p.add_argument('--lr3', type=float, default=0.001)
    p.add_argument('--reg1', type=float, default=0.005)
    p.add_argument('--reg2', type=float, default=0.015)
    p.add_argument('--reg3', type=float, default=0.015)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--shrink', type=float, default=100.0)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str,
                   default='output/svdpp_hybrid_grid_results.csv')
    args = p.parse_args()

    train_df, valid_df = read_data_df()
    model = train_hybrid(
        train_df, valid_df,
        num_factors=args.factors,
        lr1=args.lr1, lr2=args.lr2, lr3=args.lr3,
        reg1=args.reg1, reg2=args.reg2, reg3=args.reg3,
        k=args.k, shrink=args.shrink,
        n_epochs=args.epochs, seed=args.seed
    )

    # (Optionally dump final grid‐search summary to args.output)

if __name__ == '__main__':
    main()
