import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Helper Functions ---
from helper_functions import (
    read_data_df,
    evaluate,
    make_submission,
    get_dataset
)

NUM_EPOCHS = 3


class NeuralCollaborativeFilteringModel(nn.Module):
    """
    Neural Collaborative Filtering Class
    The architecture can be consulted at https://arxiv.org/abs/1708.05031
    """

    def __init__(self, num_scientists: int, num_papers: int, dim: int, hidden_dims=(32, 16)):
        super().__init__()

        # Assign to each scientist and paper an embedding
        self.scientist_emb_mlp = nn.Embedding(num_scientists, dim)
        self.paper_emb_mlp = nn.Embedding(num_papers, dim)

        # generate separate embeddings for the Generalised Matrix Factorisation
        self.scientist_vec_gmf = nn.Embedding(num_scientists, dim)
        self.paper_vec_gmf = nn.Embedding(num_papers, dim)

        # MLP layers
        mlp1_layers = []
        input_dim = dim * 2  # because we concatenate two embeddings

        for hdim in hidden_dims:
            mlp1_layers.append(nn.Linear(input_dim, hdim))
            mlp1_layers.append(nn.ReLU())
            mlp1_layers.append(nn.Dropout(0.2))
            input_dim = hdim

        self.mlp1 = nn.Sequential(*mlp1_layers)

        self.sigmoid = nn.Sigmoid()

        # Final prediction layer
        output_layers = []
        output_layers.append(nn.Linear(dim + hidden_dims[-1], 1))
        output_layers.append(nn.ReLU())
        self.output_layer = nn.Sequential(*output_layers)

    def forward(self, sid: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            sid: [B,], int
            pid: [B,], int

        Outputs: [B,], float
        """

        # Fetch gmf embeddings
        scientist_vec_gmf = self.scientist_vec_gmf(sid)  # [B, dim]
        paper_vec_gmf = self.paper_vec_gmf(pid)  # [B, dim]

        # compute gmf as an outer product of embeddings
        gmf = scientist_vec_gmf * paper_vec_gmf

        # Fetch mlp embeddings
        scientist_vec = self.scientist_emb_mlp(sid)  # [B, dim]
        paper_vec = self.paper_emb_mlp(pid)  # [B, dim]

        # Concatenate embeddings
        x = torch.cat([scientist_vec, paper_vec], dim=-1)  # [B, 2*dim]

        # Feed through MLP
        x = self.mlp1(x)  # [B, hdims[-1]]

        x = torch.cat([x, gmf], dim=-1)  # [B, dim+hdims[-1]]

        # Final output
        x = self.output_layer(x)  # [B, 1]

        x = x.squeeze(-1)  # [B]

        return x


def train_model(model, train_loader, valid_loader):
    """
        Trains an NCF model of NUM_EPOCH epochs
        returns: trained model
    """
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(NUM_EPOCHS):
        print("Epoch", epoch)
        # Train model for an epoch
        total_loss = 0.0
        total_data = 0
        model.train()
        for sid, pid, ratings in train_loader:
            # Move data to GPU
            sid = sid.to(device)
            pid = pid.to(device)
            ratings = ratings.to(device)

            # Make prediction and compute loss
            pred = model(sid, pid)
            loss = F.mse_loss(pred, ratings)

            # Compute gradients w.r.t. loss and take a step in that direction
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Keep track of running loss
            total_data += len(sid)
            total_loss += len(sid) * loss.item()

        # Evaluate model on validation data
        total_val_mse = 0.0
        total_val_data = 0
        model.eval()
        for sid, pid, ratings in valid_loader:
            # Move data to GPU
            sid = sid.to(device)
            pid = pid.to(device)
            ratings = ratings.to(device)

            # Clamp predictions in [1,5], since all ground-truth ratings are ints in [1,5]
            pred = model(sid, pid).clamp(1, 5)
            mse = F.mse_loss(pred, ratings)

            # Keep track of running metrics
            total_val_data += len(sid)
            total_val_mse += len(sid) * mse.item()

        print(
            f"[Epoch {epoch + 1}/{NUM_EPOCHS}] Train loss={total_loss / total_data:.3f}, Valid RMSE={(total_val_mse / total_val_data) ** 0.5:.3f}")
    return model


if __name__ == "__main__":
    # run on a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}\n")

    train_scores = []
    val_scores = []
    # train an NCF model for each random seed and compute their average accuracy
    for s in [10, 15, 20, 42, 50]:
        print(f"Seed: {s}")
        # process data
        train_df, valid_df = read_data_df(seed=s)
        train_dataset = get_dataset(train_df)
        valid_dataset = get_dataset(valid_df)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)
        print("Done processing data")

        # Define model (10k scientists, 1k papers, 32-dimensional embeddings)
        model = NeuralCollaborativeFilteringModel(10_000, 1_000, 32).to(device)

        model = train_model(model, train_loader, valid_loader)

        pred_fn = lambda sids, pids: model(torch.from_numpy(sids).to(device), torch.from_numpy(pids).to(device)).clamp(1, 5).cpu().numpy()
        # Evaluate on validation data
        with torch.no_grad():
            train_score = evaluate(train_df, pred_fn)
            val_score = evaluate(valid_df, pred_fn)

        print(f"Train RMSE: {train_score:.3f}, Validation RMSE: {val_score:.3f}")
        print("\n")
        train_scores.append(train_score)
        val_scores.append(val_score)

    val_mean_rmse = np.mean(val_scores)
    val_std_rmse = np.std(val_scores)
    train_mean_rmse = np.mean(train_scores)
    train_std_rmse = np.std(train_scores)
    print(f'''Mean train RMSE: {train_mean_rmse:.4f},
              Std train RMSE: {train_std_rmse:.4f}''')
    print(f'''Mean validation RMSE: {val_mean_rmse:.4f},
              Std validation RMSE: {val_std_rmse:.4f}''')

    # uncomment to make a submission using the last model

    # with torch.no_grad():
    #     make_submission(pred_fn, "collab-filtering-NCF.csv")
