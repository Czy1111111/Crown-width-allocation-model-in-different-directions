import os, random, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import optuna


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)


df = pd.read_excel(r'F:\1\1.xlsx')
df = df[df['TYPE'] == 0].copy()
target_col = 'SCW'
df = df[~df[target_col].isna()].copy()

embed_cols = ['DIR', 'TYPE_nearest1', 'TYPE_nearest2']
embed_col_dims = {'DIR': 4, 'TYPE_nearest1': 21, 'TYPE_nearest2': 21}

feature_cols = [
    'DBH', 'H', 'UBH', 'CW',
    'DIS1', 'HR1', 'DBHR1', 'SR1',
    'DBH_nearest1', 'Height_nearest1', 'Crown_width_nearest1',
    'DIS2', 'HR2', 'DBHR2', 'SR2',
    'DBH_nearest2', 'Height_nearest2', 'Crown_width_nearest2',
    'PV', 'PH'
]

df_filled = pd.DataFrame(); mask_cols = []
for col in feature_cols:
    is_na = df[col].isna().astype(int)
    fill_val = -1 if any(x in col for x in ['DIS', 'HR', 'DBHR', 'SR', 'Crown_width']) else 0
    df_filled[col] = df[col].fillna(fill_val)
    mask_col = f'{col}_valid'; mask_cols.append(mask_col)
    df_filled[mask_col] = 1 - is_na

for col in embed_cols:
    df_filled[col] = df[col].fillna(0).astype(int)

X_moe = np.concatenate([np.load('X_train_moe.npy'), np.load('X_val_moe.npy'), np.load('X_test_moe.npy')], axis=0)
y_moe = np.concatenate([np.load('y_train_moe.npy'), np.load('y_val_moe.npy'), np.load('y_test_moe.npy')], axis=0)
assert len(df_filled) == len(X_moe)

for prefix in ['1', '2']:
    for f in ['DBH_nearest', 'Height_nearest', 'Crown_width_nearest']:
        name = f'{f}{prefix}'
        df_filled[f'{name}_weighted'] = df_filled[f'{name}'] / (df_filled[f'DIS{prefix}'] + 0.1)

scaler = QuantileTransformer(output_distribution='normal', random_state=42)
numerical_cols = feature_cols + mask_cols + [
    'DBH_nearest1_weighted', 'DBH_nearest2_weighted',
    'Height_nearest1_weighted', 'Height_nearest2_weighted',
    'Crown_width_nearest1_weighted', 'Crown_width_nearest2_weighted'
]
X_orig_scaled = scaler.fit_transform(df_filled[numerical_cols].values)
X_all = np.hstack([X_orig_scaled, X_moe])
embed_features = df_filled[embed_cols].values.astype(int)
y_all = df[target_col].values


X_train_val, X_test, y_train_val, y_test, embed_train_val, embed_test = train_test_split(
    X_all, y_all, embed_features, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val, embed_train, embed_val = train_test_split(
    X_train_val, y_train_val, embed_train_val, test_size=0.17647, random_state=42)

def to_tensor(x, is_target=False):
    t = torch.tensor(x, dtype=torch.float32 if not is_target else torch.float32)
    return t.unsqueeze(1) if not is_target else t.unsqueeze(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_tensor = to_tensor(X_train); X_val_tensor = to_tensor(X_val); X_test_tensor = to_tensor(X_test)
y_train_tensor = to_tensor(y_train, is_target=True)
y_val_tensor = to_tensor(y_val, is_target=True)
y_test_tensor = to_tensor(y_test, is_target=True)
embed_train_tensor = torch.tensor(embed_train, dtype=torch.long)
embed_val_tensor = torch.tensor(embed_val, dtype=torch.long)
embed_test_tensor = torch.tensor(embed_test, dtype=torch.long)


class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class AttentionMLPWithEmbedding(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, dropout=0.2, act_name='swish'):
        super().__init__()
        self.embed_layers = nn.ModuleList([
            nn.Embedding(embed_col_dims['DIR'], 4),
            nn.Embedding(embed_col_dims['TYPE_nearest1'], 6),
            nn.Embedding(embed_col_dims['TYPE_nearest2'], 6),
        ])
        self.embed_out_dim = sum([4, 6, 6])

        self.input_proj = nn.Linear(input_dim + self.embed_out_dim, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)


        if act_name == 'swish':
            act_fn = Swish()
        elif act_name == 'relu':
            act_fn = nn.ReLU()
        elif act_name == 'gelu':
            act_fn = nn.GELU()
        else:
            act_fn = Swish()

        self.mlp_branch = nn.Sequential(
            nn.Linear(input_dim + self.embed_out_dim, 256),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            act_fn,
            nn.Linear(64, 32)
        )

        self.output = nn.Sequential(
            nn.Linear(d_model + 32, 64),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, embed_x):
        embed_out = [layer(embed_x[:, i]) for i, layer in enumerate(self.embed_layers)]
        embed_cat = torch.cat(embed_out, dim=-1)
        x_combined = torch.cat([x.squeeze(1), embed_cat], dim=-1).unsqueeze(1)

        x_proj = self.input_proj(x_combined)
        attn_out, _ = self.attn(x_proj, x_proj, x_proj)
        attn_out = self.norm(attn_out + x_proj).mean(dim=1)

        x_mlp = self.mlp_branch(x_combined.squeeze(1))
        x_cat = torch.cat([attn_out, x_mlp], dim=-1)
        return self.output(x_cat)


class MixedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    def forward(self, pred, target):
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.mae(pred, target)

def evaluate_r2(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return r2_score(y_true[mask], y_pred[mask]) if np.sum(mask) > 0 else -np.inf


def objective(trial):
    d_model = trial.suggest_categorical('d_model', [64, 96, 128, 192])
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 5e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    alpha = trial.suggest_float('alpha', 0.4, 0.9)
    act_name = trial.suggest_categorical('act_name', ['swish', 'relu', 'gelu'])
    optimizer_name = trial.suggest_categorical('optimizer', ['adamw', 'adam', 'sgd'])

    model = AttentionMLPWithEmbedding(X_train.shape[1], d_model, nhead, dropout, act_name).to(device)

    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    loss_fn = MixedLoss(alpha=alpha)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor, embed_train_tensor), batch_size=batch_size, shuffle=True)

    best_val_r2 = -np.inf
    best_state = None
    patience_counter = 0
    max_patience = 30

    for epoch in range(200):
        model.train()
        train_losses = []
        for xb, yb, eb in train_loader:
            xb, yb, eb = xb.to(device), yb.to(device), eb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb, eb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor.to(device), embed_val_tensor.to(device)).cpu().numpy().flatten()
            val_r2 = evaluate_r2(y_val_tensor.cpu().numpy().flatten(), val_pred)
            val_loss = loss_fn(torch.tensor(val_pred).unsqueeze(-1), y_val_tensor).item()

        scheduler.step()

        trial.report(val_r2, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()


        if val_r2 > best_val_r2 + 1e-5:
            best_val_r2 = val_r2
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                break

        if epoch % 10 == 0 or epoch == 199:
            print(f"Trial {trial.number} Epoch {epoch}: Train loss {np.mean(train_losses):.4f}, Val R2 {val_r2:.4f}")

    torch.save(best_state, 'best_attention_model.pth')
    return best_val_r2


study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20))
study.optimize(objective, n_trials=50, timeout=None)

print("BEST:", study.best_params)
print("VAL R²: %.4f" % study.best_value)


best_model = AttentionMLPWithEmbedding(
    X_train.shape[1],
    d_model=study.best_params['d_model'],
    nhead=study.best_params['nhead'],
    dropout=study.best_params['dropout'],
    act_name=study.best_params['act_name']
).to(device)

best_model.load_state_dict(torch.load("best_attention_model.pth"))
best_model.eval()
with torch.no_grad():
    test_pred = best_model(X_test_tensor.to(device), embed_test_tensor.to(device)).cpu().numpy().flatten()
    test_r2 = evaluate_r2(y_test_tensor.cpu().numpy().flatten(), test_pred)
print("TEST R²: %.4f" % test_r2)

