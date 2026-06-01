"""
LTNtorch × CNF 数据集 v3 —— 修复 collapse 版
核心改进：
  1. 混合损失 = LTN逻辑损失 + 加权BCE监督损失（解决 collapse）
  2. TF-IDF 特征替代 BOW（提升特征区分度）
  3. 每个谓词独立优化，监督不被规则稀释
  4. 动态阈值评估（非固定0.5）
"""
import os, re, pickle, random, sys
import numpy as np
from collections import defaultdict, Counter
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────
# 数值安全
# ─────────────────────────────────────────────
EPS = 1e-7

def safe(x):
    return torch.clamp(x, EPS, 1.0 - EPS)

# ─────────────────────────────────────────────
# LTN 核心（Product 语义，全程 safe）
# ─────────────────────────────────────────────
class LTNVariable:
    def __init__(self, name, values):
        self.name = name
        self.value = values

class LTNPredicate(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        v = x.value if isinstance(x, LTNVariable) else x
        return safe(torch.sigmoid(self.net(v).squeeze(-1)))  # (batch,1)→(batch,)

def f_and(u, v):    return safe(u * v)
def f_or(u, v):     return safe(u + v - u * v)
def f_not(u):       return safe(1.0 - u)
def f_impl(u, v):   return safe(1.0 - u + u * v)

def forall(u, p=2.0):
    """pMean-Error，数值稳定，沿 batch 维聚合"""
    comp  = safe(1.0 - u)
    inner = comp.pow(p).mean().clamp(min=EPS)
    return safe(1.0 - inner.pow(1.0 / p))

def sat_agg(*vals):
    """算术平均聚合（稳定）"""
    return safe(torch.stack(list(vals)).mean())

# ─────────────────────────────────────────────
# 网络结构
# ─────────────────────────────────────────────
def make_net(in_dim, hidden=256):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),         # LayerNorm 不依赖 batch size，更稳定
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(hidden, 128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(128, 1),
    )

# ─────────────────────────────────────────────
# 数据
# ─────────────────────────────────────────────
def find_data_dir():
    for c in ['CNF数据集', 'CNF#U6570#U636e#U96c6', '.']:
        if os.path.exists(os.path.join(c, 'node_dict.pkl')):
            return c
    import glob
    hits = glob.glob('**/node_dict.pkl', recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError("找不到 node_dict.pkl")

def load_raw(data_dir):
    with open(os.path.join(data_dir, 'node_dict.pkl'), 'rb') as f:
        node_dict = pickle.load(f)
    with open(os.path.join(data_dir, 'statements'), 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    print(f"[数据] 符号表={len(node_dict)}, 公理数={len(lines)}")
    return node_dict, lines

def parse(lines):
    records = []
    for i, line in enumerate(lines):
        m = re.match(r'fof\((\w+),\s*\w+,', line)
        if not m:
            continue
        name = m.group(1)
        syms = set(re.findall(r'([a-z][a-z0-9_]*)\(', line)) - {'fof'}
        # 去掉 skolem 项
        syms = {s for s in syms if not re.match(r'sk\d+$', s)}
        rels  = [s for s in syms if re.match(r'r\d+_', s)]
        attrs = [s for s in syms if re.match(r'v\d+_', s)]
        funcs = [s for s in syms if re.match(r'k\d+_', s)]
        records.append({
            'idx': i, 'name': name, 'syms': syms,
            'rels': rels, 'attrs': attrs, 'funcs': funcs,
            'n_syms': len(syms),
            'trivial': '$true' in line,
            'commut':  name.startswith('commutativity'),
        })
    return records

def build_tfidf(records, top_k=300):
    """
    TF-IDF 特征：比纯 BOW 更有区分度。
    每条公理 = 文档，符号 = 词。
    """
    # 统计 DF（文档频率）
    freq = Counter()
    for r in records:
        freq.update(s for s in r['syms'] if not s.startswith('sk'))

    # 选 top_k 高频但排除 skolem
    vocab = [s for s, _ in freq.most_common(top_k * 2)
             if not s.startswith('sk')][:top_k]
    s2i = {s: i for i, s in enumerate(vocab)}
    N   = len(records)
    D   = len(vocab)

    # TF 矩阵
    X = np.zeros((N, D), dtype=np.float32)
    for i, r in enumerate(records):
        for s in r['syms']:
            if s in s2i:
                X[i, s2i[s]] += 1.0

    # IDF
    df  = (X > 0).sum(axis=0) + 1.0
    idf = np.log((N + 1.0) / df) + 1.0
    X   = X * idf

    # L2 归一化
    norms = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
    X = X / norms

    print(f"[特征] TF-IDF 维度={D}, 非零均值={float((X>1e-6).mean()):.3f}")
    return X, vocab, s2i

def build_labels(records):
    n   = len(records)
    med = np.median([r['n_syms'] for r in records])
    SET_RELS  = {'r2_hidden', 'r1_tarski', 'r1_xboole_0'}
    EMPTY_SYM = {'v1_xboole_0', 'k1_xboole_0'}

    L = {k: np.zeros(n, np.float32) for k in
         ['theorem','definition','empty','setrel','commut','functor','manysym']}
    for i, r in enumerate(records):
        nm = r['name']
        L['theorem'][i]    = 1. if re.match(r't\d+_', nm) else 0.
        L['definition'][i] = 1. if re.match(r'd\d+_', nm) else 0.
        L['empty'][i]      = 1. if r['syms'] & EMPTY_SYM else 0.
        L['setrel'][i]     = 1. if set(r['rels']) & SET_RELS else 0.
        L['commut'][i]     = 1. if r['commut'] else 0.
        L['functor'][i]    = 1. if r['funcs'] else 0.
        L['manysym'][i]    = 1. if r['n_syms'] > med else 0.

    print("\n[标签分布]")
    for k, v in L.items():
        print(f"  {k:<12}: 正例率={v.mean():.1%}")
    return L

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class AxiomDS(Dataset):
    def __init__(self, X, L):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.L = {k: torch.tensor(v, dtype=torch.float32) for k, v in L.items()}
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.X[i], {k: v[i] for k, v in self.L.items()}

def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]}
    return xs, ys

# ─────────────────────────────────────────────
# 知识库
# ─────────────────────────────────────────────
class KB(nn.Module):
    """
    混合损失 = α·LTN规则损失 + (1-α)·加权BCE监督损失
    α=0.3：规则约束占30%，监督信号占70%
    这样规则满足度仍然有意义，但监督信号不会被淹没。
    """
    def __init__(self, d, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        h = 256
        self.P = nn.ModuleDict({
            'theorem':    LTNPredicate(make_net(d, h)),
            'definition': LTNPredicate(make_net(d, h)),
            'empty':      LTNPredicate(make_net(d, h)),
            'setrel':     LTNPredicate(make_net(d, h)),
            'commut':     LTNPredicate(make_net(d, h)),
            'functor':    LTNPredicate(make_net(d, h)),
            'manysym':    LTNPredicate(make_net(d, h)),
        })

    def _bce_supervised(self, pred, label):
        """加权 BCE：正例权重 = min(负/正, 30)"""
        pos = label.sum().clamp(min=1.)
        neg = (1 - label).sum().clamp(min=1.)
        pw  = (neg / pos).clamp(max=30.)
        return F.binary_cross_entropy(pred, label,
               weight=torch.where(label > 0.5,
                      torch.full_like(label, pw.item()),
                      torch.ones_like(label)))

    def _ltn_supervised(self, pred, label):
        """LTN 形式的监督：∀x label→P ∧ ¬label→¬P"""
        ph = forall(f_impl(label, pred))
        nh = forall(f_impl(f_not(label), f_not(pred)))
        return sat_agg(ph, nh)

    def forward(self, xb, labels):
        xv   = LTNVariable("x", xb)
        pval = {k: p(xv) for k, p in self.P.items()}

        # ── BCE 监督损失（主导项）──
        bce_total = sum(self._bce_supervised(pval[k], labels[k])
                        for k in pval) / len(pval)

        # ── LTN 规则满足度 ──
        # R1: EmptySet → ¬Theorem
        r1 = forall(f_impl(pval['empty'], f_not(pval['theorem'])))
        # R2: SetRelation → ¬Definition
        r2 = forall(f_impl(pval['setrel'], f_not(pval['definition'])))
        # R3: Commutative → HasFunctor
        r3 = forall(f_impl(pval['commut'], pval['functor']))
        # R4: ManySymbols → Theorem
        r4 = forall(f_impl(pval['manysym'], pval['theorem']))
        # R5: Definition → ¬Theorem
        r5 = forall(f_impl(pval['definition'], f_not(pval['theorem'])))

        ltn_sat = sat_agg(r1, r2, r3, r4, r5)
        ltn_loss = 1.0 - ltn_sat

        # ── 混合损失 ──
        loss = self.alpha * ltn_loss + (1.0 - self.alpha) * bce_total

        info = {
            'loss': loss.item(), 'bce': bce_total.item(),
            'ltn_sat': ltn_sat.item(), 'ltn_loss': ltn_loss.item(),
            'R1': r1.item(), 'R2': r2.item(), 'R3': r3.item(),
            'R4': r4.item(), 'R5': r5.item(),
        }
        return loss, info

# ─────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────
def evaluate(kb, loader, device):
    kb.eval()
    all_pred = defaultdict(list)
    all_true = defaultdict(list)

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            xv = LTNVariable("x", xb)
            for k, p in kb.P.items():
                scores = p(xv).cpu().numpy()
                labels = yb[k].numpy()
                all_pred[k].extend(scores.tolist())
                all_true[k].extend(labels.tolist())

    print(f"\n{'谓词':<14}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}"
          f"{'AUC':>7}{'正例/总':>10}")
    print("-" * 65)

    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

    for k in kb.P:
        scores = np.array(all_pred[k])
        trues  = np.array(all_true[k]).astype(int)
        total  = len(trues)
        pos    = trues.sum()

        # 动态阈值：ROC 最优点
        if pos > 0 and pos < total:
            from sklearn.metrics import roc_curve
            fpr, tpr, thr = roc_curve(trues, scores)
            j = np.argmax(tpr - fpr)
            best_thr = thr[j]
            auc = roc_auc_score(trues, scores)
        else:
            best_thr, auc = 0.5, 0.5

        preds = (scores >= best_thr).astype(int)
        acc   = (preds == trues).mean()
        prec  = precision_score(trues, preds, zero_division=0)
        rec   = recall_score(trues, preds, zero_division=0)
        f1    = f1_score(trues, preds, zero_division=0)

        print(f"{k:<14}{acc:>7.3f}{prec:>7.3f}{rec:>7.3f}{f1:>7.3f}"
              f"{auc:>7.3f}{pos:>5}/{total}")

def eval_rules(kb, loader, device):
    kb.eval()
    sats = defaultdict(list)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = {k: v.to(device) for k, v in yb.items()}
            _, info = kb(xb, yb)
            for k in ['R1','R2','R3','R4','R5','ltn_sat']:
                sats[k].append(info[k])

    rule_desc = {
        'R1': 'EmptySet → ¬Theorem',
        'R2': 'SetRel   → ¬Definition',
        'R3': 'Commut   → HasFunctor',
        'R4': 'ManySym  → Theorem',
        'R5': 'Def      → ¬Theorem',
        'ltn_sat': '总规则满足度',
    }
    print(f"\n{'规则':<10}{'描述':<26}{'满足度':>8}")
    print("-" * 48)
    for k, desc in rule_desc.items():
        v = np.mean(sats[k])
        bar = '█' * int(v * 20)
        print(f"{k:<10}{desc:<26}{v:>8.4f}  {bar}")

# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────
def train(data_dir, epochs=50, bs=128, lr=3e-4, alpha=0.3, device='cpu'):
    print("=" * 65)
    print("  LTNtorch × CNF — 混合损失版（BCE监督 + LTN规则）")
    print("=" * 65)

    _, lines = load_raw(data_dir)
    records  = parse(lines)
    X, vocab, s2i = build_tfidf(records, top_k=300)
    labels   = build_labels(records)

    n = len(records)
    idx = list(range(n)); random.seed(42); random.shuffle(idx)
    sp  = int(0.8 * n)
    ti, vi = idx[:sp], idx[sp:]

    def sub(arr, ii):
        if isinstance(arr, np.ndarray): return arr[ii]
        return {k: v[ii] for k, v in arr.items()}

    tr_ds = AxiomDS(X[ti], sub(labels, ti))
    te_ds = AxiomDS(X[vi], sub(labels, vi))
    tr_dl = DataLoader(tr_ds, bs, shuffle=True, collate_fn=collate, drop_last=True)
    te_dl = DataLoader(te_ds, bs, shuffle=False, collate_fn=collate)
    print(f"\n训练={len(tr_ds)}, 测试={len(te_ds)}, 特征维度={X.shape[1]}")

    kb   = KB(X.shape[1], alpha=alpha).to(device)
    opt  = torch.optim.AdamW(kb.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(tr_dl))

    print(f"\n混合损失权重：α(LTN)={alpha}  (1-α)(BCE)={1-alpha}")
    print(f"\n{'Ep':>4}|{'Loss':>7}|{'BCE':>7}|{'LTNsat':>7}"
          f"|{'R1':>7}|{'R2':>7}|{'R3':>7}|{'R4':>7}|{'R5':>7}")
    print("-" * 68)

    history = []
    for ep in range(1, epochs + 1):
        kb.train()
        agg = defaultdict(float); nb = 0
        for xb, yb in tr_dl:
            xb = xb.to(device)
            yb = {k: v.to(device) for k, v in yb.items()}
            opt.zero_grad()
            loss, info = kb(xb, yb)
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(kb.parameters(), 1.0)
            opt.step(); sched.step()
            for k, v in info.items(): agg[k] += v
            nb += 1
        if nb == 0: continue
        g = lambda k: agg[k] / nb
        history.append({'ep': ep, 'loss': g('loss'), 'bce': g('bce'), 'sat': g('ltn_sat')})

        if ep % 5 == 0 or ep == 1:
            print(f"{ep:>4}|{g('loss'):>7.4f}|{g('bce'):>7.4f}|{g('ltn_sat'):>7.4f}"
                  f"|{g('R1'):>7.4f}|{g('R2'):>7.4f}|{g('R3'):>7.4f}"
                  f"|{g('R4'):>7.4f}|{g('R5'):>7.4f}")

    print("\n" + "=" * 65)
    print("测试集评估（动态阈值，ROC 最优点）")
    print("=" * 65)
    evaluate(kb, te_dl, device)

    print("\n规则满足度（测试集）")
    eval_rules(kb, te_dl, device)

    print(f"\n最终: loss={history[-1]['loss']:.4f} "
          f"| bce={history[-1]['bce']:.4f} "
          f"| ltn_sat={history[-1]['sat']:.4f}")
    return kb, history

if __name__ == '__main__':
    data_dir = find_data_dir()
    print(f"数据目录: {data_dir}\n")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(data_dir, epochs=50, bs=128, lr=3e-4, alpha=0.3, device=device)