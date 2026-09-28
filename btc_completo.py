# =====================================================================
# Rede Bitcoin pela regra de admissão por idade: todos os observáveis
#
#  backbone = massa com mais ligações (definição do orientador)
#  (1) casca e lei de graus do backbone       -> ref. sintética rho=0,043
#  (2) fração fermiónica dos satélites        -> ref. sintética rho=0,043
#  (3) nulidade N-2nu, por massa e por faixa
#  (4) ponte: casca (arestas e pernas) vs nulidade
#
#  casca em ARESTAS (arestas-folha) para a estatística;
#  casca em PERNAS (nós de grau 1 na massa) para a ponte com a nulidade.
# =====================================================================
import numpy as np, pandas as pd, time
from collections import defaultdict, deque

FICHEIRO = "bq_BTC_results.csv"
t0 = time.time()

# ---------------- 1. dados: pares únicos pela ordem da 1.ª ocorrência
df = pd.read_csv(FICHEIRO, usecols=["src", "dst"])
df = df.dropna(subset=["src", "dst"]); df = df[df.src != df.dst]
cod = pd.factorize(pd.concat([df.src, df.dst]), sort=False)[0]
n = len(df); I, J = cod[:n], cod[n:]
vistos = set(); ar = []
for u, v in zip(I, J):
    p = (u, v) if u < v else (v, u)
    if p not in vistos:
        vistos.add(p); ar.append((u, v))
print(f"transações {n:,} | arestas únicas {len(ar):,}  ({time.time()-t0:.0f}s)")

# ---------------- 2. regra de admissão por idade
pdo = defaultdict(set); pat = []
for u, v in ar:
    d = pdo[u] ^ pdo[v]
    if d:
        m = min(d); nv = v if m in pdo[u] else u
        pdo[nv].add(m); pat[m].append((u, v))
    else:
        m = len(pat); pat.append([(u, v)]); pdo[u].add(m); pdo[v].add(m)
print(f"massas {len(pat):,}  ({time.time()-t0:.0f}s)")

# ---------------- 3. observáveis por massa
def por_massa(arestas):
    adj = defaultdict(list)
    for u, v in arestas: adj[u].append(v); adj[v].append(u)
    deg = {x: len(adj[x]) for x in adj}
    N, E = len(adj), len(arestas)
    casca_ar = sum(1 for u, v in arestas if deg[u] == 1 or deg[v] == 1)
    pernas = sum(1 for x in deg if deg[x] == 1)
    d = dict(deg); vivo = set(adj)                     # emparelhamento máximo
    fila = deque(x for x in adj if d[x] == 1); nu = 0  # (poda de folhas)
    while fila:
        x = fila.popleft()
        if x not in vivo or d[x] != 1: continue
        y = next((w for w in adj[x] if w in vivo), None)
        if y is None: vivo.discard(x); continue
        nu += 1; vivo.discard(x); vivo.discard(y)
        for w in adj[y]:
            if w in vivo:
                d[w] -= 1
                if d[w] == 1: fila.append(w)
    return N, E, casca_ar, pernas, nu, deg

tab = np.zeros((len(pat), 5), dtype=np.int64)          # N, E, casca_ar, pernas, nu
bb = max(range(len(pat)), key=lambda i: len(pat[i]))   # mais ligações
for i, a in enumerate(pat):
    N, E, c, p, nu, deg = por_massa(a)
    tab[i] = (N, E, c, p, nu)
    if i == bb: deg_bb = np.array(list(deg.values()))
print(f"observáveis calculados  ({time.time()-t0:.0f}s)\n")
N_, E_, C_, P_, NU_ = tab.T
NUL_ = N_ - 2 * NU_

# ---------------- 4. resultados
print("=== (1) BACKBONE = massa com mais ligações ===")
print(f"  N = {N_[bb]:,}  E = {E_[bb]:,}  (fração da rede: {N_[bb]/len(pdo):.4f})")
print(f"  casca / arestas = {C_[bb]/E_[bb]:.4f}   "
      f"ref. sintética rho=0,043: 0,670 (preferencial) | 0,500 (uniforme)")
for k, ba, un in ((1, 2/3, 1/2), (2, 1/6, 1/4), (3, 1/15, 1/8)):
    print(f"  P({k}) = {(deg_bb == k).mean():.4f}   BA {ba:.4f} | uniforme {un:.4f}")
print(f"  grau máximo = {deg_bb.max():,} ({deg_bb.max()/N_[bb]:.3f} dos nós)")

sat = np.arange(len(pat)) != bb
ferm = (C_ % 2 == 1)
print("\n=== (2) FRAÇÃO FERMIÓNICA dos satélites (casca em arestas) ===")
print(f"  todos os satélites : {ferm[sat].mean():.4f}   "
      f"ref. sintética rho=0,043: 0,806 ± 0,016")
print(f"  excluindo N=2      : {ferm[sat & (N_ > 2)].mean():.4f}")

faixas = [("2", N_ == 2), ("3-9", (N_ >= 3) & (N_ < 10)),
          ("10-99", (N_ >= 10) & (N_ < 100)),
          ("100-999", (N_ >= 100) & (N_ < 1000)), (">=1000", N_ >= 1000)]
print("\n=== (3)-(4) POR FAIXA DE TAMANHO ===")
print(f"  {'faixa':>8} {'massas':>9} {'casca/E':>8} {'ferm.':>7} {'nul/N':>7} "
      f"{'casca/nul':>10} {'pernas/nul':>11}")
for nome, s in faixas:
    if s.sum() == 0: continue
    nul = NUL_[s].sum()
    r1 = C_[s].sum()/nul if nul else float("inf")
    r2 = P_[s].sum()/nul if nul else float("inf")
    print(f"  {nome:>8} {s.sum():>9,} {C_[s].sum()/E_[s].sum():8.4f} "
          f"{ferm[s].mean():7.4f} {nul/N_[s].sum():7.4f} {r1:10.2f} {r2:11.2f}")

print("\n=== AGREGADOS ===")
print(f"  nulidade/N (soma sobre massas) : {NUL_.sum()/N_.sum():.4f}   (ref. 0,40)")
print(f"  pernas >= nulidade em todas?   : {bool((P_ >= NUL_).all())}")
print(f"  casca/nulidade | pernas/nulidade (N>=3): "
      f"{C_[N_>2].sum()/NUL_[N_>2].sum():.3f} | {P_[N_>2].sum()/NUL_[N_>2].sum():.3f}")

np.savez("observaveis_por_massa.npz", N=N_, E=E_, casca=C_, pernas=P_, nu=NU_, bb=bb)
print(f"\nguardado: observaveis_por_massa.npz   (total {time.time()-t0:.0f}s)")
