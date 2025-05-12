import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import rdchem
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

# --------------------------------------------------------------------
# 1) Levenshtein and Closeness
# --------------------------------------------------------------------
def levenshtein_distance(s1, s2):
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    dp = [[i if j == 0 else j if i == 0 else 0 for j in range(len(s2) + 1)]
          for i in range(len(s1) + 1)]
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
    return dp[-1][-1]

def closeness(dist, L=5):
    return (1 - dist / float(L)) * 100

# --------------------------------------------------------------------
# 2) Descriptor Extraction
# --------------------------------------------------------------------
def extract_descriptors(smi):
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return np.zeros(12, dtype=float)
    total = mol.GetNumAtoms()
    carb   = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    hetero = total - carb
    single = sum(1 for b in mol.GetBonds() if b.GetBondType() == rdchem.BondType.SINGLE)
    double = sum(1 for b in mol.GetBonds() if b.GetBondType() == rdchem.BondType.DOUBLE)
    arom   = sum(1 for b in mol.GetBonds() if b.GetIsAromatic())
    wedge  = sum(1 for b in mol.GetBonds()
                 if b.GetBondDir() in (rdchem.BondDir.ENDUPRIGHT, rdchem.BondDir.ENDDOWNRIGHT))
    ri     = mol.GetRingInfo()
    nr     = ri.NumRings()
    sizes  = [len(r) for r in ri.AtomRings()] or [0]
    avg_r  = sum(sizes) / len(sizes)
    rs     = sum(1 for a in mol.GetAtoms() if a.HasProp('_CIPCode'))
    ch     = Chem.GetFormalCharge(mol)
    vec = np.array([carb, hetero, single, double, arom, wedge,
                    nr, avg_r, rs, ch, total, 0.0], dtype=float)
    vec[:10] /= max(total, 1)
    return vec

# --------------------------------------------------------------------
# 3) Model: Policy and Multi-Head Critic
# --------------------------------------------------------------------
class PolicyCritic(nn.Module):
    def __init__(self, in_dim, hidden=512, drop=0.2, critics=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop),
        )
        self.policy = nn.Linear(hidden, 5)
        self.critics = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(critics)])

    def forward(self, x):
        h      = self.shared(x)
        logits = torch.clamp(self.policy(h), -10.0, 10.0)
        vals   = torch.stack([c(h).squeeze(-1) for c in self.critics], dim=1)
        mean_v = vals.mean(dim=1)
        std_v  = vals.std(dim=1)
        return logits, mean_v, std_v

# --------------------------------------------------------------------
# 4) Plackett–Luce log-likelihood and Sampling
# --------------------------------------------------------------------
def pl_loglik(logits, perm, temp=1.0):
    rem, ll = list(range(5)), 0.0
    for ch in perm:
        idx    = rem.index(int(ch)-1)
        scores = logits[rem] / temp
        p      = torch.softmax(scores, dim=0)
        ll    += torch.log(p[idx] + 1e-9)
        rem.pop(idx)
    return ll

def sample_perm(logits, temp=1.0):
    rem, logp, perm = list(range(5)), 0.0, []
    for _ in range(5):
        scores = torch.nan_to_num(logits[rem]/temp, nan=0.0, posinf=10.0, neginf=-10.0)
        p      = torch.softmax(scores, dim=0)
        p      = torch.clamp(p, 1e-8, 1.0)
        p      = p / p.sum()
        idx    = torch.multinomial(p, 1).item()
        choice = rem.pop(idx)
        logp  += torch.log(p[idx])
        perm.append(choice + 1)
    return "".join(map(str, perm)), logp

# --------------------------------------------------------------------
# 5) Beam Search for Final Decoding
# --------------------------------------------------------------------
def beam_search(logits, k=5):
    beams = [([], 0.0, list(range(5)))]
    for _ in range(5):
        candidates = []
        for perm, score, rem in beams:
            scores = torch.nan_to_num(logits[rem], nan=0.0, posinf=10.0, neginf=-10.0)
            p      = torch.softmax(scores, dim=0).cpu().numpy()
            for i,prob in enumerate(p):
                candidates.append((perm+[rem[i]],
                                   score + math.log(max(prob,1e-9)),
                                   rem[:i] + rem[i+1:]))
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]
    best = beams[0][0]
    return "".join(str(i+1) for i in best)

# --------------------------------------------------------------------
# 6) Training Pipeline
# --------------------------------------------------------------------
def train():
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Participant data
    participants = [
        {"Participant":"p1","StringSet1":"24513","SMILES1":"CCO","StringSet2":"42153","SMILES2":"CC(=O)O","StringSet3":"53214","SMILES3":"c1ccccc1","StringSet4":"31245","SMILES4":"C1CCCCC1","StringSet5":"35124","SMILES5":"CCN"},
        {"Participant":"p2","StringSet1":"25314","SMILES1":"CNC","StringSet2":"31524","SMILES2":"CCOCC","StringSet3":"14235","SMILES3":"OCCO","StringSet4":"","SMILES4":"","StringSet5":"","SMILES5":""},
        {"Participant":"p3","StringSet1":"25413","SMILES1":"CCC","StringSet2":"52134","SMILES2":"CCCC","StringSet3":"21543","SMILES3":"CCCCC","StringSet4":"13245","SMILES4":"CCCCCC","StringSet5":"25134","SMILES5":"CCCCCCC"},
        {"Participant":"p4","StringSet1":"25413","SMILES1":"C1=CC=CC=C1","StringSet2":"43125","SMILES2":"C1CC1","StringSet3":"32514","SMILES3":"C1CCC1","StringSet4":"14235","SMILES4":"C1CCCC1","StringSet5":"25134","SMILES5":"C1CCCCC1"},
        {"Participant":"p5","StringSet1":"23514","SMILES1":"CCN(CC)CC","StringSet2":"12543","SMILES2":"CNCN","StringSet3":"14235","SMILES3":"CCCl","StringSet4":"","SMILES4":"","StringSet5":"","SMILES5":""},
        {"Participant":"p6","StringSet1":"15234","SMILES1":"CCCCO","StringSet2":"45132","SMILES2":"CCCO","StringSet3":"21435","SMILES3":"CCOCC","StringSet4":"23154","SMILES4":"C1COCC1","StringSet5":"32145","SMILES5":"COC"},
        {"Participant":"p7","StringSet1":"15324","SMILES1":"C1=CC=CN=C1","StringSet2":"54312","SMILES2":"C1CCNCC1","StringSet3":"12435","SMILES3":"CC(=O)NC","StringSet4":"15234","SMILES4":"CCN(C)C","StringSet5":"","SMILES5":""},
        {"Participant":"p8","StringSet1":"42351","SMILES1":"CCOCCO","StringSet2":"31542","SMILES2":"OCCOC","StringSet3":"","SMILES3":"","StringSet4":"","SMILES4":"","StringSet5":"","SMILES5":""},
    ]
    df = pd.DataFrame(participants)

    # Compute human data
    for i in range(1,6):
        df[f"Dist{i}"]  = df[f"StringSet{i}"].apply(lambda s:5 if len(s)!=5 else levenshtein_distance(s,"12345"))
        df[f"Close{i}"] = df[f"Dist{i}"].apply(closeness)
    df["SetsEval"] = df[[f"StringSet{i}" for i in range(1,6)]].apply(
        lambda r: sum(len(str(x))==5 for x in r), axis=1)

    # Unroll trials
    trials = []
    for _,r in df.iterrows():
        for i in range(1,6):
            perm = r[f"StringSet{i}"]
            if len(perm)==5:
                entry = {"Participant":r.Participant, "Set":i, "HumanPerm":perm}
                for j in range(1,6):
                    entry[f"SMILES{j}"] = r[f"SMILES{j}"]
                trials.append(entry)
    trials = pd.DataFrame(trials)

    # Train/Val split
    train_df = trials.sample(frac=0.8, random_state=0)
    val_df   = trials.drop(train_df.index)

    def build_X(df_):
        feats=[]
        for _,r in df_.iterrows():
            vecs=[extract_descriptors(r[f"SMILES{j}"]) for j in range(1,6)]
            feats.append(np.concatenate(vecs))
        return torch.tensor(np.stack(feats), dtype=torch.float32)

    X_train = build_X(train_df).to(device)
    X_val   = build_X(val_df).to(device)
    y_train = train_df.HumanPerm.tolist()
    y_val   = val_df.HumanPerm.tolist()

    # Model, optimizer, schedulers
    model   = PolicyCritic(in_dim=60, hidden=512, drop=0.2, critics=2).to(device)
    scaler  = GradScaler()
    sup_epochs, rl_epochs = 1500, 500
    opt_sup = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched_sup = OneCycleLR(opt_sup, max_lr=1e-3, total_steps=sup_epochs)
    opt_rl  = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-6)
    sched_rl = CosineAnnealingWarmRestarts(opt_rl, T_0=100, T_mult=2)

    writer = SummaryWriter(f"runs/{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs("results", exist_ok=True)

    # Supervised Pretraining
    for ep in range(1, sup_epochs+1):
        model.train(); opt_sup.zero_grad()
        logits, _, _ = model(X_train)
        loss_sup     = -torch.stack([pl_loglik(logits[i], y_train[i], temp=0.8)
                                     for i in range(len(y_train))]).mean()
        loss_sup.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_sup.step(); sched_sup.step()
        if ep % 200 == 0:
            model.eval()
            with torch.no_grad():
                val_nll = -torch.stack([pl_loglik(model(X_val)[0][i], y_val[i], temp=0.8)
                                        for i in range(len(y_val))]).mean().item()
            print(f"Sup ep{ep:04d} | Train NLL {loss_sup:.3f} | Val NLL {val_nll:.3f}")

    # RL Fine-tuning
    for ep in range(1, rl_epochs+1):
        model.train(); opt_rl.zero_grad()
        with autocast():
            logits, vals, _ = model(X_train)
            batch_r, batch_d, batch_acc, logps, sup_term = [], [], [], [], 0.0
            temp = 0.5 + 0.5 * math.cos(math.pi * ep / rl_epochs)
            for i, hum in enumerate(y_train):
                perm, lp = sample_perm(logits[i], temp=temp)
                L        = levenshtein_distance(perm, hum)
                r        = math.exp(-1.0 * L)
                batch_r.append(r); batch_d.append(L); batch_acc.append(int(perm==hum))
                logps.append(lp)
                sup_term += -pl_loglik(logits[i], hum, temp=0.8)
            R   = torch.tensor(batch_r, device=device)
            ADV = R - vals
            ADV = (ADV - ADV.mean()) / (ADV.std(unbiased=False) + 1e-8)
            LP  = torch.stack(logps)
            sup_term /= len(y_train)

            policy_loss = -(ADV.detach() * LP).mean()
            value_loss  = 1.5 * nn.MSELoss()(vals, R)
            entropy     = -(torch.softmax(logits/temp,1)*torch.log_softmax(logits/temp,1)).sum(1).mean()
            loss        = policy_loss + value_loss - 0.01*entropy + 0.01*sup_term

        scaler.scale(loss).backward()
        scaler.unscale_(opt_rl)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        finite = all((p.grad is None or torch.isfinite(p.grad).all()) for p in model.parameters())
        if finite:
            scaler.step(opt_rl)
        else:
            print(f"Skipping update at ep{ep} due to non-finite gradients")
        scaler.update()
        sched_rl.step(ep + rl_epochs)  # warm restarts on global step

        if ep % 100 == 0:
            print(f"RL ep{ep:03d} | Loss {loss.item():.3f} | R {np.mean(batch_r):.3f} | D {np.mean(batch_d):.2f} | A {100*np.mean(batch_acc):.1f}%")

    # Final Evaluation
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(X_train)
    preds, ds, rs = [], [], []
    for i, lg in enumerate(logits.cpu()):
        p = beam_search(lg, k=5)
        L = levenshtein_distance(p, y_train[i])
        preds.append(p); ds.append(L); rs.append(math.exp(-1.0 * L))
    train_df["ModelPerm"]  = preds
    train_df["HumanClose"] = [closeness(d) for d in ds]
    train_df["ModelClose"] = [r*100 for r in rs]
    train_df.to_excel("results/train_results.xlsx", index=False)

    # Validation
    with torch.no_grad():
        logits_v, _, _ = model(X_val)
    preds, ds, rs = [], [], []
    for i, lg in enumerate(logits_v.cpu()):
        p = beam_search(lg, k=5)
        L = levenshtein_distance(p, y_val[i])
        preds.append(p); ds.append(L); rs.append(math.exp(-1.0 * L))
    val_df["ModelPerm"]  = preds
    val_df["HumanClose"] = [closeness(d) for d in ds]
    val_df["ModelClose"] = [r*100 for r in rs]
    val_df.to_excel("results/val_results.xlsx", index=False)

    print("Training complete. Results → `results/` directory.")


if __name__ == "__main__":
    train()
