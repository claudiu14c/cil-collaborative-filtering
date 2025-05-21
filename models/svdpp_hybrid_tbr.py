#!/usr/bin/env python3

"""
svdpp_hybrid_tbr.py

Extended hybrid model (baseline + SVD++ + neighborhood + wish-list) with per-epoch RMSE logging.

Usage example:
$ python hybrid_tbr_submit.py \
    --data_dir /cluster/courses/cil/collaborative_filtering/data \
    --factors 20 \
    --lr1 0.007 --lr2 0.007 --lr3 0.001 \
    --reg1 0.005 --reg2 0.015 --reg3 0.015 \
    --k 300 --shrink 100.0 \
    --epochs 30 \
    --seed 42
"""

import os
import argparse
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import defaultdict

# Helper functions

def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def read_data_df(data_dir):
    """Reads data and splits into train/validation sets (75/25)."""
    df = pd.read_csv(os.path.join(data_dir, "train_ratings.csv"))
    df[["sid","pid"]] = df["sid_pid"].str.split("_", expand=True)
    df = df.drop(columns=["sid_pid"])
    df["sid"] = df["sid"].astype(int)
    df["pid"] = df["pid"].astype(int)
    train_df, valid_df = train_test_split(df, test_size=0.25, random_state=0)
    return train_df, valid_df


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


def hybrid_pred_with_tbr(model, sids, pids,
                         min_rating=1.0, max_rating=5.0):
    """
    Prediction for the extended hybrid model with wish-list.
    """
    mu = model['mu']
    bu, bi = model['bu'], model['bi']
    p, q, y, z = model['p'], model['q'], model['y'], model['z']
    w, c, d = model['w'], model['c'], model['d']
    user2ind, item2ind = model['user2ind'], model['item2ind']
    implicit = model['implicit']
    wishlist = model['wishlist']
    neighbors = model['neighbors']
    ratings_by_user = model['ratings_by_user']
    f = model['num_factors']

    preds = []
    for sid, pid in zip(sids, pids):
        if sid in user2ind and pid in item2ind:
            u, i = user2ind[sid], item2ind[pid]
            # SVD++ terms
            Nu, Tw = implicit[u], wishlist[u]
            imp = y[Nu].sum(0)/np.sqrt(len(Nu)) if Nu else np.zeros(f)
            wish= z[Tw].sum(0)/np.sqrt(len(Tw)) if Tw else np.zeros(f)
            svdpp = q[i].dot(p[u] + imp + wish)
            # neighbors
            Rk = [j for j in neighbors[i] if j in ratings_by_user[u]]
            Tk = [j for j in neighbors[i] if j in wishlist[u]]
            sr = np.sqrt(len(Rk)) if Rk else 1.0
            st = np.sqrt(len(Tk)) if Tk else 1.0

            nexp = sum((ratings_by_user[u][j] - (mu + bu[u] + bi[j])) * w[i][idx_j]
                       for idx_j, j in enumerate(Rk)) / sr
            nimp = sum(c[i][idx_j] for idx_j, j in enumerate(Rk)) / sr
            nwish= sum(d[i][idx_j] for idx_j, j in enumerate(Tk)) / st

            pred = mu + bu[u] + bi[i] + svdpp + nexp + nimp + nwish
        else:
            pred = mu
        preds.append(np.clip(pred, min_rating, max_rating))

    return np.array(preds, dtype=np.float32)


def train_hybrid_with_tbr(train_df, wish_df, valid_df=None,
                          num_factors=20,
                          lr1=0.007, lr2=0.007, lr3=0.001,
                          reg1=0.005, reg2=0.015, reg3=0.015,
                          k=300, shrink=100.0,
                          n_epochs=30, seed=42):
    """
    Train extended hybrid model with wish-list, logging RMSE each epoch to:
      output/hybrid_tbr_curve_f{f}_lr{lr1}-{lr2}-{lr3}_reg{reg1}-{reg2}-{reg3}.csv
    """
    np.random.seed(seed)
    os.makedirs('output', exist_ok=True)

    log_file = (f"output/hybrid_tbr_curve_f{num_factors}"
                f"_lr{lr1}-{lr2}-{lr3}"
                f"_reg{reg1}-{reg2}-{reg3}.csv")
    with open(log_file, 'w') as f:
        f.write("epoch,rmse\n")

    # 1) Build unified user/item ID lists
    if isinstance(wish_df, set):
        w_sids, w_pids = zip(*wish_df)
        w_sids = np.array(w_sids, dtype=int)
        w_pids = np.array(w_pids, dtype=int)
    else:
        w_sids = wish_df['sid'].values.astype(int)
        w_pids = wish_df['pid'].values.astype(int)

    all_sids = np.unique(np.concatenate([train_df['sid'].values, w_sids]))
    all_pids = np.unique(np.concatenate([train_df['pid'].values, w_pids]))
    user2ind = {sid: u for u, sid in enumerate(all_sids)}
    item2ind = {pid: i for i, pid in enumerate(all_pids)}
    n_users, n_items = len(all_sids), len(all_pids)

    # 2) Index arrays for training ratings
    u_arr = train_df['sid'].map(user2ind).values.astype(np.int32)
    i_arr = train_df['pid'].map(item2ind).values.astype(np.int32)
    r_arr = train_df['rating'].values.astype(np.float32)

    # 3) Baseline biases
    mu = r_arr.mean()
    bu = np.zeros(n_users, np.float32)
    bi = np.zeros(n_items, np.float32)
    for _ in range(10):
        for u, i, r in zip(u_arr, i_arr, r_arr):
            e = r - (mu + bu[u] + bi[i])
            bu[u] += lr1 * (e - reg1 * bu[u])
            bi[i] += lr1 * (e - reg1 * bi[i])
            
    # 4) Standard implicit feedback (rated items)
    implicit = defaultdict(list)
    for u, i in zip(u_arr, i_arr):
        implicit[u].append(i)
    Nu_list = [np.array(implicit[u], dtype=np.int32) for u in range(n_users)]
    Nu_count = np.array([len(Nu_list[u]) for u in range(n_users)], dtype=np.int32)
    sqrt_Nu = np.where(Nu_count>0, np.sqrt(Nu_count, dtype=np.float32), 1.0)

    # 5) Wish-list feedback
    wishlist = defaultdict(list)
    if isinstance(wish_df, set):
        pairs = wish_df
    else:
        pairs = zip(wish_df['sid'], wish_df['pid'])
    for sid, pid in pairs:
        if sid in user2ind and pid in item2ind:
            wishlist[user2ind[sid]].append(item2ind[pid])
    Tw_list = [np.array(wishlist[u], dtype=np.int32) for u in range(n_users)]
    Tw_count = np.array([len(Tw_list[u]) for u in range(n_users)], dtype=np.int32)
    sqrt_Tw = np.where(Tw_count>0, np.sqrt(Tw_count, dtype=np.float32), 1.0)

    # 6) Neighborhood structure
    neighbors = compute_item_neighbors(u_arr, i_arr, r_arr,
                                       mu, bu, bi,
                                       n_items, k=k, shrink=shrink)

    # 7) Initialize latent factors & offsets
    p = np.random.normal(0,0.1,(n_users,num_factors)).astype(np.float32)
    q = np.random.normal(0,0.1,(n_items,num_factors)).astype(np.float32)
    y = np.random.normal(0,0.1,(n_items,num_factors)).astype(np.float32)
    z = np.random.normal(0,0.1,(n_items,num_factors)).astype(np.float32)
    w = {i: np.zeros(len(neighbors[i]),np.float32) for i in range(n_items)}
    c = {i: np.zeros(len(neighbors[i]),np.float32) for i in range(n_items)}
    d = {i: np.zeros(len(neighbors[i]),np.float32) for i in range(n_items)}

    # 8) Precompute sums
    y_sum = np.zeros((n_users,num_factors), np.float32)
    z_sum = np.zeros((n_users,num_factors), np.float32)
    for u in range(n_users):
        if Nu_count[u]>0:
            y_sum[u] = y[Nu_list[u]].sum(0) / sqrt_Nu[u]
        if Tw_count[u]>0:
            z_sum[u] = z[Tw_list[u]].sum(0) / sqrt_Tw[u]

    # 9) Ratings map for residual lookup
    ratings_by_user = [dict() for _ in range(n_users)]
    for u, i, r in zip(u_arr, i_arr, r_arr):
        ratings_by_user[u][i] = r

    # 10) SGD training
    n = len(r_arr)
    for epoch in range(1, n_epochs+1):
        perm = np.random.permutation(n)
        for idx in perm:
            u, i, r = u_arr[idx], i_arr[idx], r_arr[idx]
            imp = y_sum[u]
            wish = z_sum[u]
            svdpp_val = q[i].dot(p[u] + imp + wish)

            # neighbors
            Rk = [j for j in neighbors[i] if j in ratings_by_user[u]]
            Tk = [j for j in neighbors[i] if j in wishlist[u]]
            sr = np.sqrt(len(Rk)) if Rk else 1.0
            st = np.sqrt(len(Tk)) if Tk else 1.0

            nexp = sum((ratings_by_user[u][j] - (mu + bu[u] + bi[j])) * w[i][idx_j]
                       for idx_j, j in enumerate(Rk)) / sr
            nimp = sum(c[i][idx_j] for idx_j, j in enumerate(Rk)) / sr
            nwish = sum(d[i][idx_j] for idx_j, j in enumerate(Tk)) / st

            pred = mu + bu[u] + bi[i] + svdpp_val + nexp + nimp + nwish
            err  = r - pred

            # update biases
            bu[u] += lr1 * (err - reg1 * bu[u])
            bi[i] += lr1 * (err - reg1 * bi[i])

            # update factors
            old_pu = p[u].copy()
            p[u] += lr2 * (err * q[i]   - reg2 * p[u])
            q[i] += lr2 * (err * (old_pu + imp + wish) - reg2 * q[i])

            # implicit
            if Nu_count[u]>0:
                coeff = lr2 * err / sqrt_Nu[u]
                for j in Nu_list[u]:
                    y[j] += coeff * q[i] - lr2 * reg2 * y[j]
                y_sum[u] = y[Nu_list[u]].sum(0) / sqrt_Nu[u]

            # wish-list
            if Tw_count[u]>0:
                coeff = lr2 * err / sqrt_Tw[u]
                for j in Tw_list[u]:
                    z[j] += coeff * q[i] - lr2 * reg2 * z[j]
                z_sum[u] = z[Tw_list[u]].sum(0) / sqrt_Tw[u]

            # neighborhood updates
            for idx_j, j in enumerate(Rk):
                bas_res = ratings_by_user[u][j] - (mu + bu[u] + bi[j])
                w[i][idx_j] += lr3 * (err/sr * bas_res - reg3 * w[i][idx_j])
                c[i][idx_j] += lr3 * (err/sr          - reg3 * c[i][idx_j])
            for idx_j, j in enumerate(Tk):
                d[i][idx_j] += lr3 * (err/st - reg3 * d[i][idx_j])

        # end of epoch
        print(f"Epoch {epoch}/{n_epochs} completed.", flush=True)

        if valid_df is not None:
            model = dict(
                mu=mu, bu=bu, bi=bi,
                p=p, q=q, y=y, z=z,
                w=w, c=c, d=d,
                user2ind=user2ind, item2ind=item2ind,
                implicit=implicit, wishlist=wishlist,
                neighbors=neighbors, num_factors=num_factors,
                ratings_by_user=ratings_by_user
            )
            preds = hybrid_pred_with_tbr(
                model,
                valid_df['sid'].values,
                valid_df['pid'].values
            )
            rmse = root_mean_squared_error(valid_df['rating'].values, preds)
            print(f"Validation RMSE after {epoch} epochs: {rmse:.4f}", flush=True)
            with open(log_file, 'a') as f:
                f.write(f"{epoch},{rmse}\n")

    return dict(mu=mu, bu=bu, bi=bi,
                p=p, q=q, y=y, z=z,
                w=w, c=c, d=d,
                user2ind=user2ind, item2ind=item2ind,
                implicit=implicit, wishlist=wishlist,
                neighbors=neighbors, num_factors=num_factors,
                ratings_by_user=ratings_by_user)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  type=str,
                        default="/cluster/courses/cil/collaborative_filtering/data",
                        help="Directory with train_ratings.csv and train_tbr.csv")
    parser.add_argument('--factors',   type=int,   default=20)
    parser.add_argument('--lr1',       type=float, default=0.007)
    parser.add_argument('--lr2',       type=float, default=0.007)
    parser.add_argument('--lr3',       type=float, default=0.001)
    parser.add_argument('--reg1',      type=float, default=0.005)
    parser.add_argument('--reg2',      type=float, default=0.015)
    parser.add_argument('--reg3',      type=float, default=0.015)
    parser.add_argument('--k',         type=int,   default=300)
    parser.add_argument('--shrink',    type=float, default=100.0)
    parser.add_argument('--epochs',    type=int,   default=30)
    parser.add_argument('--seed',      type=int,   default=42)
    args = parser.parse_args()

    # 1) load train+validation splits
    train_df, valid_df = read_data_df(args.data_dir)

    # 2) Load TBR data and build a lookup set (as in your notebook)
    tbr_df = pd.read_csv(os.path.join(args.data_dir, "train_tbr.csv"))
    tbr_pairs = set(zip(tbr_df['sid'], tbr_df['pid']))

    # 3) call the training function with that set
    _ = train_hybrid_with_tbr(
        train_df,
        tbr_pairs,
        valid_df=valid_df,
        num_factors=args.factors,
        lr1=args.lr1, lr2=args.lr2, lr3=args.lr3,
        reg1=args.reg1, reg2=args.reg2, reg3=args.reg3,
        k=args.k,
        shrink=args.shrink,
        n_epochs=args.epochs,
        seed=args.seed
    )

if __name__ == '__main__':
    main()
