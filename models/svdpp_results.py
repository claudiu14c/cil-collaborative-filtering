#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
from typing import Callable
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from helper_functions import make_submission, read_data_df


def svdpp_pred(model, sids, pids):
    """
    Prediction function for trained SVD++ model, with correct baseline fallback.
    """
    mu = model['mu']
    user2ind = model['user2ind']
    item2ind = model['item2ind']
    b_u = model['b_u']
    b_i = model['b_i']
    p = model['p']
    q = model['q']
    y = model['y']
    num_factors = model['num_factors']
    implicit = model['implicit']

    preds = []
    for sid, pid in zip(sids, pids):
        pred = mu
        if sid in user2ind:
            u = user2ind[sid]
            pred += b_u[u]
        if pid in item2ind:
            i = item2ind[pid]
            pred += b_i[i]
        if (sid in user2ind) and (pid in item2ind):
            Nu = implicit[u]
            sqrt_Nu = np.sqrt(len(Nu)) if Nu else 1.0
            imp_sum = np.sum(y[Nu, :], axis=0) / sqrt_Nu if Nu else np.zeros(num_factors)
            pred += np.dot(q[i], p[u] + imp_sum)
        preds.append(pred)
    preds = np.array(preds, dtype=np.float32)
    return np.clip(preds, 1.0, 5.0)


def train_svdpp(
    train_df: pd.DataFrame,
    num_factors=20, lr=0.005, reg=0.02,
    n_epochs=20, seed=42
):
    """
    NumPy-only SVD++ training on full dataset (no validation).
    """
    np.random.seed(seed)

    # 1) remap IDs
    sids = train_df['sid'].unique()
    pids = train_df['pid'].unique()
    user2ind = {sid: i for i, sid in enumerate(sids)}
    item2ind = {pid: i for i, pid in enumerate(pids)}
    n_users, n_items = len(sids), len(pids)

    # 2) prepare arrays
    user_arr   = train_df['sid'].map(user2ind).to_numpy(dtype=np.int32)
    item_arr   = train_df['pid'].map(item2ind).to_numpy(dtype=np.int32)
    rating_arr = train_df['rating'].to_numpy(dtype=np.float32)

    # 3) global mean
    mu = np.float32(rating_arr.mean())

    # 4) init parameters
    b_u = np.zeros(n_users, np.float32)
    b_i = np.zeros(n_items, np.float32)
    p   = np.random.normal(0, 0.1, (n_users, num_factors)).astype(np.float32)
    q   = np.random.normal(0, 0.1, (n_items, num_factors)).astype(np.float32)
    y   = np.random.normal(0, 0.1, (n_items, num_factors)).astype(np.float32)

    # 5) implicit feedback
    implicit = {u: [] for u in range(n_users)}
    for u, i in zip(user_arr, item_arr):
        implicit[u].append(i)
    Nu_list  = [np.array(implicit[u], dtype=np.int32) for u in range(n_users)]
    Nu_count = np.array([len(a) for a in Nu_list], dtype=np.int32)
    sqrt_Nu  = np.where(Nu_count>0, np.sqrt(Nu_count, dtype=np.float32), 1.0)

    # 6) precompute y_sum
    y_sum = np.zeros((n_users, num_factors), np.float32)
    for u in range(n_users):
        if Nu_count[u]:
            y_sum[u] = y[Nu_list[u]].sum(0) / sqrt_Nu[u]

    # 7) SGD training
    n_ratings = rating_arr.shape[0]
    for epoch in range(1, n_epochs+1):
        perm = np.random.permutation(n_ratings)
        for idx in perm:
            u = user_arr[idx]; i = item_arr[idx]; r = rating_arr[idx]
            imp = y_sum[u]
            pred = mu + b_u[u] + b_i[i] + q[i].dot(p[u] + imp)
            err  = r - pred

            # update biases and factors
            b_u[u] += lr * (err - reg * b_u[u])
            b_i[i] += lr * (err - reg * b_i[i])
            p_old   = p[u].copy()
            p[u]   += lr * (err * q[i]   - reg * p[u])
            q[i]   += lr * (err * (p_old + imp) - reg * q[i])

            # implicit updates
            if Nu_count[u]:
                coeff = lr * err / sqrt_Nu[u]
                idxs  = Nu_list[u]
                yj    = y[idxs]
                y[idxs] = yj + coeff * q[i] - lr * reg * yj
                y_sum[u] = y_sum[u] + coeff * q[i] - lr * reg * y_sum[u]

        print(f"Epoch {epoch}/{n_epochs} completed.")

    return {
        'mu':mu, 'b_u':b_u, 'b_i':b_i,
        'p':p,   'q':q,   'y':y,
        'user2ind':user2ind, 'item2ind':item2ind,
        'implicit':implicit, 'num_factors':num_factors
    }


def multi_seed_evaluate(
    factors: int, lr: float, reg: float,
    epochs: int, seeds: list
):
    """
    Train and evaluate SVD++ across multiple random seeds,
    reporting mean and stddev of RMSE on train and validation splits.
    """
    train_rmses = []
    val_rmses = []

    for seed in seeds:
        print(f"\n=== Seed: {seed} ===")
        train_df, valid_df = read_data_df(seed=seed)
        print(f"Training on {len(train_df)} ratings; validating on {len(valid_df)} ratings.")

        model = train_svdpp(
            train_df,
            num_factors=factors,
            lr=lr,
            reg=reg,
            n_epochs=epochs,
            seed=seed
        )

        # Evaluate on train split
        s_train = train_df['sid'].values
        p_train = train_df['pid'].values
        y_train = train_df['rating'].values
        preds_train = svdpp_pred(model, s_train, p_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
        train_rmses.append(rmse_train)
        print(f"Train RMSE: {rmse_train:.4f}")

        # Evaluate on validation split
        s_val = valid_df['sid'].values
        p_val = valid_df['pid'].values
        y_val = valid_df['rating'].values
        preds_val = svdpp_pred(model, s_val, p_val)
        rmse_val = np.sqrt(mean_squared_error(y_val, preds_val))
        val_rmses.append(rmse_val)
        print(f"Validation RMSE: {rmse_val:.4f}")

    # Summary
    print("\n=== Summary across seeds ===")
    print(f"Mean Train RMSE: {np.mean(train_rmses):.4f}, Std: {np.std(train_rmses):.4f}")
    print(f"Mean Validation RMSE: {np.mean(val_rmses):.4f}, Std: {np.std(val_rmses):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train SVD++ and (optionally) evaluate across multiple seeds."
    )
    parser.add_argument('--factors', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--reg', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42,
                        help="Single seed for one-off training and prediction")
    parser.add_argument('--multi_seed', action='store_true',
                        help="Run train/eval over multiple seeds and report stats")
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[10, 15, 20, 42, 50],
                        help="List of seeds for multi-seed evaluation")
    args = parser.parse_args()

    job_id = os.environ.get('SLURM_JOB_ID', 'local')

    if args.multi_seed:
        multi_seed_evaluate(
            args.factors, args.lr, args.reg,
            args.epochs, args.seeds
        )
    else:
        train_df = read_data_df(split=0.0)
        # read_data_for_training()
        print(f"Training on {len(train_df)} ratings with factors={args.factors}, lr={args.lr}, reg={args.reg}, epochs={args.epochs}, seed={args.seed}")
        model = train_svdpp(
            train_df,
            num_factors=args.factors,
            lr=args.lr,
            reg=args.reg,
            n_epochs=args.epochs,
            seed=args.seed
        )

        def pred_fn(sids, pids):
            return svdpp_pred(model, sids, pids)
        # Submission generation commented out
        out_name = f"svdpp_sub_{job_id}.csv"
        print(f"Generating submission to {out_name}")
        make_submission(pred_fn, out_name)


if __name__ == '__main__':
    main()
