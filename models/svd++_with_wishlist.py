import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
import time
import argparse

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# --- Helper Functions ---
from helper_functions import (
    read_data_df,
    read_full_training_data,
    read_tbr_df,
    clip_and_make_submission,
)

def evaluate_model_predictions(true_ratings, pred_ratings):
    """Calculates RMSE after clipping predictions to [1.0, 5.0]."""
    preds_clipped = np.clip(pred_ratings, 1.0, 5.0)
    return root_mean_squared_error(true_ratings, preds_clipped)

def evaluate_with_model(model_dict, eval_df, pred_function):
    """Evaluates the model on the given dataframe using the prediction function."""
    if eval_df is None or eval_df.empty:
        return np.nan
    preds = pred_function(model_dict, eval_df["sid"].values, eval_df["pid"].values)
    return evaluate_model_predictions(eval_df["rating"].values, preds)


def plot_training_curves(n_total_epochs_run, train_rmse_hist, val_rmse_hist, title_prefix=""):
    """Plots training and validation RMSE over epochs."""
    epochs_ran = len(train_rmse_hist)
    if epochs_ran == 0:
        print(f"{title_prefix}: No training history to plot.")
        return
    epochs_range = range(1, epochs_ran + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_rmse_hist, "bo-", label="Training RMSE")
    valid_val_epochs = [e for e, r in zip(epochs_range, val_rmse_hist[:epochs_ran]) if not np.isnan(r)]
    valid_val_rmse_points = [r for r in val_rmse_hist[:epochs_ran] if not np.isnan(r)]
    if valid_val_epochs:
        plt.plot(valid_val_epochs, valid_val_rmse_points, "ro-", label="Validation RMSE")
    plt.title(f"{title_prefix} Training & Validation RMSE (Up to {epochs_ran} Epochs)")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- SVD++ with Wishlist Integration ---

def train_svdpp_with_wishlist_integrated(train_df, tbr_df, num_factors, lr, reg, n_epochs, valid_df=None, early_stopping_patience=5, evaluate_every_n_epochs=1):
    """
    Trains an SVD++ model with integrated wishlist data using stochastic gradient descent.

    Args:
        train_df: DataFrame with training ratings (sid, pid, rating).
        tbr_df: DataFrame with wishlist data (sid, pid).
        num_factors: Number of latent factors.
        lr: Learning rate for gradient updates.
        reg: Regularization parameter.
        n_epochs: Maximum number of training epochs.
        valid_df: Optional validation DataFrame for early stopping.
        early_stopping_patience: Epochs to wait for improvement before stopping.
        evaluate_every_n_epochs: Frequency of validation evaluation.

    Returns:
        dict: Best model parameters.
        list: Training RMSE history.
        list: Validation RMSE history.
    """
    # Map user and item IDs from both ratings and wishlist data
    sids_all = np.union1d(train_df["sid"].unique(), tbr_df["sid"].unique())
    pids_all = np.union1d(train_df["pid"].unique(), tbr_df["pid"].unique())
    user2ind = {sid: i for i, sid in enumerate(sids_all)}
    item2ind = {pid: i for i, pid in enumerate(pids_all)}
    n_users, n_items = len(sids_all), len(pids_all)

    # Prepare training data arrays
    train_df_mappable = train_df[train_df["sid"].isin(user2ind) & train_df["pid"].isin(item2ind)]
    user_arr = train_df_mappable["sid"].map(user2ind).to_numpy(dtype=np.int32)
    item_arr = train_df_mappable["pid"].map(item2ind).to_numpy(dtype=np.int32)
    rating_arr = train_df_mappable["rating"].to_numpy(dtype=np.float32)
    mu = np.float32(rating_arr.mean()) if len(rating_arr) > 0 else np.float32(3.5)

    # Initialize model parameters
    b_u = np.zeros(n_users, np.float32)  # User biases
    b_i = np.zeros(n_items, np.float32)  # Item biases
    p = np.random.normal(0, 0.01, (n_users, num_factors)).astype(np.float32)  # User factors
    q = np.random.normal(0, 0.01, (n_items, num_factors)).astype(np.float32)  # Item factors
    y = np.random.normal(0, 0.01, (n_items, num_factors)).astype(np.float32)  # Implicit feedback factors

    # Build combined implicit feedback (ratings + wishlist)
    implicit_combined = {u_idx: set() for u_idx in range(n_users)}
    for sid, pid in zip(train_df["sid"], train_df["pid"]):
        if sid in user2ind and pid in item2ind:
            implicit_combined[user2ind[sid]].add(item2ind[pid])
    for sid, pid in zip(tbr_df["sid"], tbr_df["pid"]):
        if sid in user2ind and pid in item2ind:
            implicit_combined[user2ind[sid]].add(item2ind[pid])

    Nu_list_combined = [np.array(list(implicit_combined[u_idx]), dtype=np.int32) for u_idx in range(n_users)]
    Nu_count_combined = np.array([len(a) for a in Nu_list_combined], dtype=np.int32)
    sqrt_Nu_combined_inv = np.where(Nu_count_combined > 0, 1.0 / np.sqrt(Nu_count_combined, dtype=np.float32), 0.0)
    sqrt_Nu_for_pred = np.where(Nu_count_combined > 0, np.sqrt(Nu_count_combined, dtype=np.float32), 1.0)

    # Training loop variables
    train_rmse_history, val_rmse_history = [], []
    best_metric, epochs_no_improve = float("inf"), 0
    best_model_params = {}
    n_ratings = len(user_arr)

    # Start training
    for epoch in range(n_epochs):
        start_time = time.time()
        squared_errors = []

        # Compute implicit feedback sum for each user
        y_sum = np.zeros((n_users, num_factors), np.float32)
        for u_idx in range(n_users):
            if Nu_count_combined[u_idx] > 0:
                y_sum[u_idx] = y[Nu_list_combined[u_idx]].sum(axis=0) / sqrt_Nu_for_pred[u_idx]

        # Stochastic gradient descent
        perm = np.random.permutation(n_ratings)
        for idx in perm:
            u, i, r = user_arr[idx], item_arr[idx], rating_arr[idx]
            imp = y_sum[u]
            pred = mu + b_u[u] + b_i[i] + q[i].dot(p[u] + imp)
            err = r - pred
            squared_errors.append(err ** 2)

            # Update parameters
            b_u[u] += lr * (err - reg * b_u[u])
            b_i[i] += lr * (err - reg * b_i[i])
            p_old = p[u].copy()
            p[u] += lr * (err * q[i] - reg * p[u])
            q[i] += lr * (err * (p_old + imp) - reg * q[i])
            if Nu_count_combined[u] > 0:
                coeff_grad_y = err * q[i] * sqrt_Nu_combined_inv[u]
                y[Nu_list_combined[u]] += lr * (coeff_grad_y - reg * y[Nu_list_combined[u]])
                y_sum[u] += lr * (err * q[i] * sqrt_Nu_combined_inv[u] - reg * y_sum[u])

        # Evaluate epoch performance
        train_rmse = np.sqrt(np.mean(squared_errors))
        train_rmse_history.append(train_rmse)
        stopping_metric = train_rmse

        if valid_df is not None and (epoch + 1) % evaluate_every_n_epochs == 0 or epoch == n_epochs - 1:
            temp_model = {"mu": mu, "b_u": b_u, "b_i": b_i, "p": p, "q": q, "y": y,
                          "user2ind": user2ind, "item2ind": item2ind,
                          "Nu_list_combined": Nu_list_combined, "sqrt_Nu_for_pred": sqrt_Nu_for_pred,
                          "num_factors": num_factors}
            val_rmse = evaluate_with_model(temp_model, valid_df, svdpp_pred_wishlist_integrated)
            val_rmse_history.append(val_rmse)
            stopping_metric = val_rmse
            print(f"Epoch {epoch+1}/{n_epochs}: Train RMSE: {train_rmse:.4f}, Valid RMSE: {val_rmse:.4f}, Time: {time.time() - start_time:.2f}s")
        else:
            val_rmse_history.append(np.nan)
            print(f"Epoch {epoch+1}/{n_epochs}: Train RMSE: {train_rmse:.4f}, Time: {time.time() - start_time:.2f}s")

        # Early stopping check
        if stopping_metric < best_metric:
            best_metric = stopping_metric
            epochs_no_improve = 0
            best_model_params = {"mu": mu, "b_u": b_u.copy(), "b_i": b_i.copy(),
                                 "p": p.copy(), "q": q.copy(), "y": y.copy(),
                                 "user2ind": user2ind, "item2ind": item2ind,
                                 "Nu_list_combined": [arr.copy() for arr in Nu_list_combined],
                                 "sqrt_Nu_for_pred": sqrt_Nu_for_pred.copy(),
                                 "num_factors": num_factors}
        else:
            epochs_no_improve += 1
            if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return best_model_params, train_rmse_history, val_rmse_history

def svdpp_pred_wishlist_integrated(model, sids_arr, pids_arr):
    """Predicts ratings using the trained SVD++ model with wishlist integration."""
    mu, user2ind, item2ind = model["mu"], model["user2ind"], model["item2ind"]
    b_u, b_i, p, q, y = model["b_u"], model["b_i"], model["p"], model["q"], model["y"]
    Nu_list_combined = model["Nu_list_combined"]
    sqrt_Nu_for_pred = model["sqrt_Nu_for_pred"]
    num_factors = model["num_factors"]

    preds = np.full(len(sids_arr), mu, dtype=np.float32)
    for k in range(len(sids_arr)):
        sid, pid = sids_arr[k], pids_arr[k]
        if sid in user2ind and pid in item2ind:
            u, i = user2ind[sid], item2ind[pid]
            imp_sum = np.sum(y[Nu_list_combined[u]], axis=0) / sqrt_Nu_for_pred[u] if Nu_list_combined[u].size > 0 else 0
            preds[k] = mu + b_u[u] + b_i[i] + np.dot(q[i], p[u] + imp_sum)
    return preds

# --- Main Execution ---

def main():
    """Main function to train SVD++ with wishlist integration and generate submission."""
    parser = argparse.ArgumentParser(description="SVD++ with Integrated Wishlist for Recommendation System")
    parser.add_argument('--factors', type=int, default=50, help="Number of latent factors")
    parser.add_argument('--lr', type=float, default=0.007, help="Learning rate")
    parser.add_argument('--reg', type=float, default=0.04, help="Regularization parameter")
    parser.add_argument('--epochs_valid', type=int, default=20, help="Epochs for validation training")
    parser.add_argument('--patience_valid', type=int, default=3, help="Early stopping patience for validation")
    parser.add_argument('--epochs_full', type=int, default=45, help="Epochs for full data training")
    parser.add_argument('--patience_full', type=int, default=5, help="Early stopping patience for full training")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load data
    train_df_split, valid_df_split = read_data_df()
    tbr_df = read_tbr_df()

    # Train with validation
    print("Training SVD++ with Wishlist (Validation Split)")
    model_val, train_hist_val, val_hist_val = train_svdpp_with_wishlist_integrated(
        train_df_split, tbr_df, args.factors, args.lr, args.reg, args.epochs_valid,
        valid_df=valid_df_split, early_stopping_patience=args.patience_valid
    )
    # plot_training_curves(args.epochs_valid, train_hist_val, val_hist_val, "SVD++ Wishlist (Validation)") # Uncomment to view the training curve on training vs validation dataset
    final_val_rmse = val_hist_val[-1] if val_hist_val else train_hist_val[-1]
    print(f"Final Validation RMSE: {final_val_rmse:.4f}")

    # Train on full data
    print("\nTraining SVD++ with Wishlist (Full Data)")
    full_train_df = read_full_training_data()
    model_full, train_hist_full, _ = train_svdpp_with_wishlist_integrated(
        full_train_df, tbr_df, args.factors, args.lr, args.reg, args.epochs_full,
        early_stopping_patience=args.patience_full
    )
    # plot_training_curves(args.epochs_full, train_hist_full, [], "SVD++ Wishlist (Full)") # Uncomment to view the training curve on full dataset
    print(f"Lowest Training RMSE: {min(train_hist_full):.4f}")

    # Generate submission
    pred_fn = lambda sids, pids: svdpp_pred_wishlist_integrated(model_full, sids, pids)
    filename = f"svdpp_wishlist_f{args.factors}_lr{args.lr}_reg{args.reg}_ep{len(train_hist_full)}.csv"
    clip_and_make_submission(pred_fn, filename)

if __name__ == "__main__":
    main()