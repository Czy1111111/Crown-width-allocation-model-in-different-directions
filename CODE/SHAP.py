# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
sns.set(style="whitegrid", context="talk")


col_name_map = {
    'DBH':'DBH','H':'TreeHeight','UBH':'HeightUnderBranch','CW':'TotalCrownWidth',
    'DIS1':'Distance1','HR1':'HeightRatio1','DBHR1':'DBHRatio1','SR1':'SlopeRatio1',
    'DBH_nearest1':'DBH_Nearest1','Height_nearest1':'Height_Nearest1','Crown_width_nearest1':'CrownWidth_Nearest1',
    'DIS2':'Distance2','HR2':'HeightRatio2','DBHR2':'DBHRatio2','SR2':'SlopeRatio2',
    'DBH_nearest2':'DBH_Nearest2','Height_nearest2':'Height_Nearest2','Crown_width_nearest2':'CrownWidth_Nearest2',
    'PV':'PV','PH':'PH'
}
embed_map = {'DIR':'Direction','TYPE_nearest1':'TypeNearest1','TYPE_nearest2':'TypeNearest2'}


df = pd.read_excel(r'F:\1\1.xlsx')
df = df[df['TYPE'] == 0].dropna(subset=['SCW'])
y_raw = df['SCW'].values

feature_cols = [
    'DBH','H','UBH','CW',
    'DIS1','HR1','DBHR1','SR1',
    'DBH_nearest1','Height_nearest1','Crown_width_nearest1',
    'DIS2','HR2','DBHR2','SR2',
    'DBH_nearest2','Height_nearest2','Crown_width_nearest2',
    'PV','PH'
]
embed_cols = ['DIR','TYPE_nearest1','TYPE_nearest2']


df_filled = pd.DataFrame(); mask_cols = []
for c in feature_cols:
    na = df[c].isna().astype(int)
    fill = -1 if any(k in c for k in ['DIS','HR','DBHR','SR','Crown_width']) else 0
    df_filled[c] = df[c].fillna(fill)
    mc = f"{c}_valid"; mask_cols.append(mc)
    df_filled[mc] = 1 - na


for c in embed_cols:
    df_filled[c] = df[c].fillna(0).astype(int)


X_moe = np.concatenate([
    np.load('X_train_moe.npy'),
    np.load('X_val_moe.npy'),
    np.load('X_test_moe.npy')
], axis=0)
assert len(df_filled) == len(X_moe)


bases = ['DBH_nearest','Height_nearest','Crown_width_nearest']
for p in ['1','2']:
    for b in bases:
        nm = f"{b}{p}"
        df_filled[f"{nm}_weighted"] = df_filled[nm] / (df_filled[f"DIS{p}"] + 0.1)


rm = {}
for c in feature_cols:
    en = col_name_map[c]
    rm[c] = en
    rm[f"{c}_valid"] = f"{en}_valid"
for b in bases:
    for p in ['1','2']:
        ch = f"{b}{p}"
        rm[f"{ch}_weighted"] = f"{col_name_map[ch]}_Weighted"
for c in embed_cols:
    rm[c] = embed_map[c]
df_filled.rename(columns=rm, inplace=True)


num_feats      = [col_name_map[c] for c in feature_cols]
mask_feats     = [f"{col_name_map[c]}_valid" for c in feature_cols]
weighted_feats = [f"{col_name_map[f'{b}{p}']}_Weighted" for b in bases for p in ['1','2']]
numerical_cols = num_feats + mask_feats + weighted_feats
embed_english  = [embed_map[c] for c in embed_cols]


scaler = QuantileTransformer(output_distribution='normal', random_state=42)
X_num  = scaler.fit_transform(df_filled[numerical_cols].values)
X_all  = np.hstack([X_num, X_moe])
embed_feats = df_filled[embed_english].values.astype(int)


X_trval, X_test, y_trval, y_test, e_trval, e_test = train_test_split(
    X_all, y_raw, embed_feats, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val, e_train, e_val = train_test_split(
    X_trval, y_trval, e_trval, test_size=0.17647, random_state=42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def to_tensor(x, is_target=False):
    t = torch.tensor(x, dtype=torch.float32)
    return t.unsqueeze(-1) if is_target else t.unsqueeze(1)

X_tr, X_v, X_te = to_tensor(X_train), to_tensor(X_val), to_tensor(X_test)
y_tr, y_v, y_te = to_tensor(y_train, True), to_tensor(y_val, True), to_tensor(y_test, True)
e_tr, e_v, e_te = (torch.tensor(e_train, dtype=torch.long),
                   torch.tensor(e_val,   dtype=torch.long),
                   torch.tensor(e_test,  dtype=torch.long))


class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class AttentionMLPWithEmbedding(nn.Module):
    def __init__(self, in_dim, d_model=96, nhead=4, dropout=0.1178, act_name='gelu'):
        super().__init__()
        self.embed_layers = nn.ModuleList([
            nn.Embedding(4,4), nn.Embedding(21,6), nn.Embedding(21,6)
        ])
        self.emb_dim = 4+6+6
        self.proj    = nn.Linear(in_dim + self.emb_dim, d_model)
        self.attn    = nn.MultiheadAttention(d_model, nhead,
                                             batch_first=True, dropout=dropout)
        self.norm    = nn.LayerNorm(d_model)
        af = {'swish':Swish(), 'relu':nn.ReLU(), 'gelu':nn.GELU()}[act_name]
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + self.emb_dim,256), af, nn.Dropout(dropout),
            nn.Linear(256,64), af, nn.Linear(64,32)
        )
        self.out = nn.Sequential(
            nn.Linear(d_model+32,64), af, nn.Dropout(dropout),
            nn.Linear(64,1)
        )
    def forward(self, x, e):
        embs = [lay(e[:,i]) for i,lay in enumerate(self.embed_layers)]
        cat  = torch.cat(embs, dim=-1)
        xcat = torch.cat([x.squeeze(1), cat], dim=-1).unsqueeze(1)
        p    = self.proj(xcat)
        a,_  = self.attn(p, p, p)
        h    = self.norm(a + p).mean(1)
        m    = self.mlp(xcat.squeeze(1))
        return self.out(torch.cat([h, m], dim=-1))


class MixedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha, self.mse, self.mae = alpha, nn.MSELoss(), nn.L1Loss()
    def forward(self, pred, tgt):
        return self.alpha*self.mse(pred, tgt) + (1-self.alpha)*self.mae(pred, tgt)

def eval_r2(y, p): return r2_score(y, p)

model   = AttentionMLPWithEmbedding(in_dim=X_train.shape[1],
                                    d_model=96, nhead=4,
                                    dropout=0.1178, act_name='gelu').to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
loss_fn = MixedLoss(alpha=0.7)
loader  = DataLoader(TensorDataset(X_tr, y_tr, e_tr), batch_size=32, shuffle=True)

best_r2, history, patience = -np.inf, [], 0
for epoch in range(200):
    model.train()
    for xb, yb, eb in loader:
        xb, yb, eb = xb.to(device), yb.to(device), eb.to(device)
        loss_fn(model(xb, eb), yb).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step(); optimizer.zero_grad()
    scheduler.step()
    model.eval()
    with torch.no_grad():
        vp = model(X_v.to(device), e_v.to(device)).cpu().numpy().flatten()
    r2 = eval_r2(y_val, vp); history.append(r2)
    if r2 > best_r2 + 1e-5:
        best_r2, best_state, patience = r2, model.state_dict(), 0
    else:
        patience += 1
        if patience >= 30:
            break
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Val R²={r2:.4f}")

torch.save(best_state, 'best_model.pth')


model.load_state_dict(torch.load('best_model.pth')); model.eval()
with torch.no_grad():
    tp = model(X_te.to(device), e_te.to(device)).cpu().numpy().flatten()
test_r2 = eval_r2(y_test, tp)
print(f"Best Val R²: {best_r2:.4f}, Test R²: {test_r2:.4f}")

plt.figure()
plt.plot(history, color='C1')
plt.xlabel('Epoch'); plt.ylabel('Validation R²')
plt.title('Validation R² over Epochs'); plt.grid(); plt.tight_layout(); plt.show()


shap_X        = np.hstack([X_num, embed_feats])
feature_names = numerical_cols + embed_english
moe_mean      = X_moe.mean(axis=0)

def model_fn(x: np.ndarray) -> np.ndarray:
    num = x[:, :len(numerical_cols)]
    emb = x[:, len(numerical_cols):].astype(int)
    b   = num.shape[0]
    full= np.hstack([num, np.tile(moe_mean, (b,1))])
    ft  = torch.tensor(full, dtype=torch.float32).unsqueeze(1).to(device)
    et  = torch.tensor(emb, dtype=torch.long).to(device)
    with torch.no_grad():
        return model(ft, et).cpu().numpy().flatten()

explainer = shap.Explainer(model_fn, shap_X[:200])
shap_vals  = explainer(shap_X[:200])


shap.summary_plot(shap_vals.values, shap_X[:200],
                  feature_names=feature_names, show=False)
plt.title('SHAP Summary: Global Importance')
plt.tight_layout(); plt.show()


idx_dbh = feature_names.index('DBH')
shap.dependence_plot(idx_dbh, shap_vals.values, shap_X[:200],
                     feature_names=feature_names, show=False)
plt.title('SHAP Dependence: DBH')
plt.tight_layout(); plt.show()


idx_dir = feature_names.index('Direction')
shap.dependence_plot(idx_dir, shap_vals.values, shap_X[:200],
                     feature_names=feature_names, show=False)
plt.title('SHAP Dependence: Direction')
plt.tight_layout(); plt.show()


mean_abs = np.abs(shap_vals.values).mean(axis=0)
top_idx  = np.argsort(mean_abs)[::-1][:10]
top_names= [feature_names[i] for i in top_idx]
top_vals = mean_abs[top_idx]

plt.figure(figsize=(6,4))
sns.barplot(x=top_vals, y=top_names,
            palette=sns.color_palette("mako", len(top_names)))
plt.title('Mean |SHAP| for Top 10 Features')
plt.xlabel('Mean |SHAP value|'); plt.ylabel('Feature')
plt.tight_layout(); plt.show()


exp0 = shap.Explanation(values=shap_vals.values[0],
                        base_values=shap_vals.base_values[0],
                        data=shap_X[0],
                        feature_names=feature_names)
shap.plots.waterfall(exp0, show=False)
plt.title('SHAP Waterfall: Sample 0')
plt.tight_layout(); plt.show()

