#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from helper_functions import read_data_df


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


def train_svdpp(train_df, valid_df=None, num_factors=20, lr=0.005, reg=0.02,
                     n_epochs=20, eval_interval=5, seed=42):
    """
    NumPy-only SVD++ training with per-epoch validation and CSV logging.
    """
    np.random.seed(seed)

    # set up per-epoch RMSE logging
    if valid_df is not None:
        os.makedirs('output', exist_ok=True)
        log_file = f"output/learning_curve_f{num_factors}_lr{lr}_reg{reg}.csv"
        with open(log_file, 'w') as logf:
            logf.write("epoch,rmse\n")

    # 1) remap IDs to 0…N–1
    sids = train_df['sid'].unique()
    pids = train_df['pid'].unique()
    user2ind = {sid: i for i, sid in enumerate(sids)}
    item2ind = {pid: i for i, pid in enumerate(pids)}
    n_users, n_items = len(sids), len(pids)

    # 2) build index arrays once
    user_arr   = train_df['sid'].map(user2ind).to_numpy(dtype=np.int32)
    item_arr   = train_df['pid'].map(item2ind).to_numpy(dtype=np.int32)
    rating_arr = train_df['rating'].to_numpy(dtype=np.float32)

    # 3) global mean as float32
    mu = np.float32(rating_arr.mean())

    # 4) init parameters (float32)
    b_u = np.zeros(n_users, np.float32)
    b_i = np.zeros(n_items, np.float32)
    p   = np.random.normal(0, 0.1, (n_users, num_factors)).astype(np.float32)
    q   = np.random.normal(0, 0.1, (n_items, num_factors)).astype(np.float32)
    y   = np.random.normal(0, 0.1, (n_items, num_factors)).astype(np.float32)

    # 5) build implicit lists
    implicit = {u: [] for u in range(n_users)}
    for u, i in zip(user_arr, item_arr):
        implicit[u].append(i)
    Nu_list  = [np.array(implicit[u], dtype=np.int32) for u in range(n_users)]
    Nu_count = np.array([len(a) for a in Nu_list], dtype=np.int32)
    sqrt_Nu  = np.where(Nu_count>0, np.sqrt(Nu_count, dtype=np.float32), 1.0)

    # 6) precompute y_sum[u] = sum_j y[j] / sqrt_Nu[u]
    y_sum = np.zeros((n_users, num_factors), np.float32)
    for u in range(n_users):
        if Nu_count[u]:
            y_sum[u] = y[Nu_list[u]].sum(0) / sqrt_Nu[u]

    # 7) SGD loop with per-epoch RMSE logging
    n_ratings = rating_arr.shape[0]
    for epoch in range(n_epochs):
        perm = np.random.permutation(n_ratings)
        for idx in perm:
            u = user_arr[idx]; i = item_arr[idx]; r = rating_arr[idx]
            imp = y_sum[u]
            pred = mu + b_u[u] + b_i[i] + q[i].dot(p[u] + imp)
            err  = r - pred

            # biases & factors
            b_u[u] += lr * (err - reg * b_u[u])
            b_i[i] += lr * (err - reg * b_i[i])
            p_old   = p[u].copy()
            p[u]   += lr * (err * q[i]   - reg * p[u])
            q[i]   += lr * (err * (p_old + imp) - reg * q[i])

            # fast implicit update
            if Nu_count[u]:
                coeff = lr * err / sqrt_Nu[u]
                idxs  = Nu_list[u]
                yj    = y[idxs]
                y[idxs] = yj + coeff * q[i] - lr * reg * yj
                y_sum[u] = y_sum[u] + lr * err * q[i] - lr * reg * y_sum[u]

        # end of epoch
        epoch_num = epoch + 1
        print(f"Epoch {epoch_num}/{n_epochs} completed.")

        # always evaluate & log
        if valid_df is not None:
            model_state = {
                'mu': mu, 'b_u': b_u, 'b_i': b_i,
                'p': p, 'q': q, 'y': y,
                'user2ind': user2ind, 'item2ind': item2ind,
                'implicit': implicit, 'num_factors': num_factors
            }
            preds = svdpp_pred(
                model_state,
                valid_df['sid'].values,
                valid_df['pid'].values
            )
            rmse = root_mean_squared_error(valid_df['rating'].values, preds)
            print(f"Validation RMSE after {epoch_num} epochs: {rmse:.4f}")
            # append to CSV
            with open(log_file, 'a') as logf:
                logf.write(f"{epoch_num},{rmse}\n")

    return {
        'mu':mu, 'b_u':b_u, 'b_i':b_i,
        'p':p,   'q':q,   'y':y,
        'user2ind':user2ind, 'item2ind':item2ind,
        'implicit':implicit, 'num_factors':num_factors
    }


def main():
    parser = argparse.ArgumentParser(description="SVD++ grid search on validation RMSE")
    parser.add_argument('--data_dir', type=str,
                        default="/cluster/courses/cil/collaborative_filtering/data",
                        help='Directory with train_ratings.csv')
    parser.add_argument('--factors', type=int, nargs='+', default=[50],
                        help='Latent factor sizes to try')
    parser.add_argument('--lrs', type=float, nargs='+', default=[0.005],
                        help='Learning rates to try')
    parser.add_argument('--regs', type=float, nargs='+', default=[0.05],
                        help='Regularization strengths to try')
    parser.add_argument('--epochs', type=int, nargs='+', default=[20],
                        help='Epoch counts to try')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='svdpp_grid_results.csv',
                        help='CSV file to save RMSE results')
    args = parser.parse_args()

    # load data
    train_df, valid_df = read_data_df(data_dir=args.data_dir)

    results = []
    total_combinations = (len(args.factors) * len(args.lrs) *
                          len(args.regs) * len(args.epochs))
    print(f"Running grid search over {total_combinations} combinations...")

    for nf in args.factors:
        for lr in args.lrs:
            for reg in args.regs:
                for ne in args.epochs:
                    print(f"Training with factors={nf}, lr={lr}, reg={reg}, epochs={ne}")
                    model = train_svdpp(
                        train_df,
                        valid_df=valid_df,
                        num_factors=nf,
                        lr=lr,
                        reg=reg,
                        n_epochs=ne,
                        seed=args.seed
                    )
                    # final evaluation
                    preds = svdpp_pred(
                        model,
                        valid_df['sid'].values,
                        valid_df['pid'].values
                    )
                    rmse = root_mean_squared_error(valid_df['rating'].values, preds)
                    print(f"--> Final RMSE: {rmse:.4f}\n")
                    results.append({
                        'num_factors': nf,
                        'lr': lr,
                        'reg': reg,
                        'epochs': ne,
                        'rmse': rmse
                    })

    # save grid-search results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"Grid search complete. Results saved to {args.output}")


if __name__ == '__main__':
    main()
