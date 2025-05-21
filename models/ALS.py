from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from helper_functions import (
    read_data_df,
    read_full_data_matrix,
    evaluate,
    make_submission,
)


class ALSRecommender:
    def __init__(self,
                 train_df: pd.DataFrame,
                 valid_df: pd.DataFrame,
                 num_factors: int = 10,
                 num_iters: int = 10,
                 reg: float = 0.1):
        """
        num_factors: latent dimensionality (k)
        num_iters: number of ALS rounds
        reg: regularization λ
        """
        # build rating matrix R, mask M
        R, user_ids, item_ids = read_full_data_matrix(train_df)
        self.M = (~np.isnan(R)).astype(np.float32)
        self.R_filled = np.nan_to_num(R, nan=0.0)
        self.n_users, self.n_items = R.shape

        # lookup tables to map sid and pid to matrix indices
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.user_index = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_index = {iid: idx for idx, iid in enumerate(item_ids)}

        # store validation DF to report validation RMSE
        self.valid_df = valid_df

        # ALS hyperparams
        self.k = num_factors
        self.num_iters = num_iters
        self.reg = reg

        # initialize latent factors (UV^{T} will approximate R)
        self.U = np.random.normal(scale=0.1, size=(self.n_users, self.k))
        self.V = np.random.normal(scale=0.1, size=(self.n_items, self.k))

    def fit(self):
        I_k = np.eye(self.k)

        for it in range(self.num_iters):
            # Fix V, optimize U
            # Solving:
            #    min_{u_i} \sum (R_{ij} - u_i^{T} v_j)^2 + \lambda ||u_i||^{2},
            # which leads to setting:
            #    u_i = (V_i^{T} V_i + \lambda I)^{-1} V_{i}^{T} r_i
            # where r_i is the vector of ratings given by person i
            for i in range(self.n_users):
                m_i = self.M[i]                   # mask of scientits i
                V_i = self.V[m_i == 1]            # papers rated by scientits i
                r_i = self.R_filled[i, m_i == 1]  # ratings given by scien. i
                A = V_i.T @ V_i + self.reg * I_k  # A = V_i^{T} V_i + \lambda I
                b = V_i.T @ r_i                   # V_{i}^{T} r_i
                self.U[i] = np.linalg.solve(A, b)

            # Fix U, optimize V
            for j in range(self.n_items):
                m_j = self.M[:, j]
                U_j = self.U[m_j == 1]
                r_j = self.R_filled[m_j == 1, j]
                A = U_j.T @ U_j + self.reg * I_k
                b = U_j.T @ r_j
                self.V[j] = np.linalg.solve(A, b)

            # train/validation RMSE
            preds_train = self.predict(train_df["sid"].values,
                                       train_df["pid"].values)
            train_rmse = root_mean_squared_error(train_df["rating"],
                                                 preds_train)
            val_rmse = evaluate(self.valid_df, self.get_pred_fn())

            print(f"Iter {it+1:2d}/{self.num_iters} — "
                  f"Train RMSE: {train_rmse:.4f}  |  "
                  f"Validate RMSE: {val_rmse:.4f}")

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        preds = []
        # get mean predicted ratings
        global_mean = np.nanmean(self.U @ self.V.T)
        for sid, pid in zip(sids, pids):
            # verify that sid and pid exist in our approximated rating matrix
            if sid in self.user_index and pid in self.item_index:
                u = self.U[self.user_index[sid]]
                v = self.V[self.item_index[pid]]
                # mathematically, return \hat{R}_{ij} = u_i^{T} v_j
                preds.append(u.dot(v))
            # otherwise, give mean to avoid error
            else:
                print("user or paper not found!")
                preds.append(global_mean)

        return np.array(preds)

    def get_pred_fn(self):
        return lambda sids, pids: self.predict(sids, pids)


val_rmses = []
train_rmses = []
for s in [10, 15, 20, 42, 50]:
    print(f"Seed: {s}")
    train_df, valid_df = read_data_df(s, 0.25)
    k = 15
    num_iterations = 30
    regParam = 20.0
    als = ALSRecommender(train_df, valid_df,
                         num_factors=k,
                         num_iters=num_iterations,
                         reg=regParam)
    als.fit()
    val_rmse = evaluate(valid_df, als.get_pred_fn())
    test_rmse = evaluate(train_df, als.get_pred_fn())
    val_rmses.append(val_rmse)
    train_rmses.append(test_rmse)
    print('\n')
    print('\n')

val_mean_rmse = np.mean(val_rmses)
val_std_rmse = np.std(val_rmses)
train_mean_rmse = np.mean(train_rmses)
train_std_rmse = np.std(train_rmses)
print(f'''Mean train RMSE: {train_mean_rmse:.4f},
          Std train RMSE: {train_std_rmse:.4f}''')
print(f'''Mean validation RMSE: {val_mean_rmse:.4f},
          Std validation RMSE: {val_std_rmse:.4f}''')


# train_df, valid_df = read_data_df()
# k = 15
# num_iterations = 30
# regParam = 20.0
# als = ALSRecommender(train_df, valid_df,
#                      num_factors=k,
#                      num_iters=num_iterations,
#                      reg=regParam)
# als.fit()
# print('\n')

# pred_fn = als.get_pred_fn()
# make_submission(pred_fn,
#                 filename=f'''submissions/als_simple_tuned_{k}_{regParam}_{num_iterations}.csv''')


def hyperparameter_search(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    param_grid: Dict[str, Any]
) -> Tuple[Dict[str, Any], float]:
    """
    Brute-force grid search over hyperparameters.

    param_grid keys: 'num_factors', 'reg', 'alpha', 'num_iters'
    Returns best_params dict and best validation RMSE.
    """
    best_rmse = float('inf')
    best_params = {}
    for k in param_grid.get('num_factors', [10]):
        for reg in param_grid.get('reg', [0.1]):
            for iters in param_grid.get('num_iters', [10]):
                print(f"Testing k={k}, reg={reg}, iters={iters}")
                model = ALSRecommender(
                    train_df, valid_df,
                    num_factors=k, num_iters=iters, reg=reg)
                model.fit()
                rmse = evaluate(valid_df, model.get_pred_fn())
                print("\n\n")
                if rmse < 0.86:
                    make_submission(
                      model.get_pred_fn(),
                      filename=f"submissions/als_simple_{k}_{reg}_{iters}.csv"
                    )
                if rmse < best_rmse:
                    best_rmse, best_params = rmse, {
                        'num_factors': k,
                        'reg': reg,
                        'num_iters': iters
                    }
    print(f"Best params: {best_params}, Valid RMSE: {best_rmse:.4f}")
    return best_params, best_rmse


# train_df, valid_df = read_data_df()
# grid = {'num_factors': [15, 20, 40, 60, 80, 100, 120],
#         'reg': [5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
#         'num_iters': [30]}
# hyperparameter_search(train_df, valid_df, grid)
