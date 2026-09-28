# =====================================================================
# Holonomia SU(2) dos ciclos exportados — rede real e modelo nulo
#
#  Um ciclo fechado da rede traça, no embedding, um polígono esférico
#  de direções. O ângulo sólido Omega que ele encerra dá a fase do
#  spinor, exp(-i Omega/2). Num ciclo de três arestas os deslocamentos
#  somam zero, logo são coplanares, o polígono degenera num grande
#  círculo e Omega = 2pi exatamente: a holonomia é -1. Para ciclos
#  maiores, o desvio de Omega face a 2pi mede a não planaridade.
#
#  As massas são árvores e não têm ciclos: estes vivem na rede original,
#  e são os que a decomposição exporta. É por isso necessário um
#  embedding global, o da componente gigante, cujos primeiros modos
#  estão quase degenerados --- limitação a declarar.
# =====================================================================
import numpy as np, pandas as pd, networkx as nx, time
from collections import deque, Counter
from scipy.sparse.linalg import eigsh

FICHEIRO = "bq_BTC_results.csv"
LMAX, MAXC, SEED = 8, 4000, 0
t0 = time.time()

# ---------------- rede
df = pd.read_csv(FICHEIRO, usecols=["src", "dst"])
df = df.dropna(subset=["src", "dst"]); df = df[df.src != df.dst]
cod = pd.factorize(pd.concat([df.src, df.dst]), sort=False)[0]
n = len(df)
G = nx.Graph(); G.add_edges_from(zip(cod[:n], cod[n:]))
GC = G.subgraph(max(nx.connected_components(G), key=len)).copy()
print(f"componente gigante: N={GC.number_of_nodes():,} E={GC.number_of_edges():,} "
      f"beta_1={GC.number_of_edges()-GC.number_of_nodes()+1:,}  ({time.time()-t0:.0f}s)")

def embed(X, d=3):
    L = nx.laplacian_matrix(X).astype(float).tocsc(); N = X.number_of_nodes()
    w = None
    for sig in (-1e-8, -1e-6, -1e-4):
        try: w, V = eigsh(L, k=d+3, sigma=sig, which="LM"); break
        except Exception: w = None
    if w is None: return None, None
    o = np.argsort(w); w, V = np.clip(w[o], 0, None), V[:, o]
    nz = [j for j in range(len(w)) if w[j] > 1e-12
          and abs(V[:, j].sum())/np.sqrt(N) < 1e-6]
    if len(nz) < d: return None, None
    return {x: V[i, nz[:d]] for i, x in enumerate(X.nodes())}, w[nz[:d]]

def ciclos_curtos(X, Lmax=LMAX, maxn=MAXC, seed=SEED):
    """Ciclos fundamentais de uma árvore BFS, via ancestral comum."""
    raiz = max(X.degree, key=lambda t: t[1])[0]
    pai = {raiz: None}; prof = {raiz: 0}; fila = deque([raiz])
    while fila:
        x = fila.popleft()
        for y in X[x]:
            if y not in pai: pai[y] = x; prof[y] = prof[x]+1; fila.append(y)
    extra = [(u, v) for u, v in X.edges()
             if pai.get(u) != v and pai.get(v) != u and u in pai and v in pai]
    rng = np.random.default_rng(seed); rng.shuffle(extra)
    out = []
    for u, v in extra:
        a, b = u, v; ca, cb = [a], [b]
        while prof[a] > prof[b]: a = pai[a]; ca.append(a)
        while prof[b] > prof[a]: b = pai[b]; cb.append(b)
        while a != b:
            a, b = pai[a], pai[b]
            if a is None or b is None: break
            ca.append(a); cb.append(b)
        if a != b: continue
        c = ca + cb[-2::-1]
        if 3 <= len(c) <= Lmax: out.append(c)
        if len(out) >= maxn: break
    return out

def angulo_solido(dirs):
    k = len(dirs)
    if k < 3: return None
    S = 0.0
    for i in range(k):
        a, b, c = dirs[i-1], dirs[i], dirs[(i+1) % k]
        ta = a - (a@b)*b; tc = c - (c@b)*b
        na, nc = np.linalg.norm(ta), np.linalg.norm(tc)
        if na < 1e-9 or nc < 1e-9: return None
        ang = np.arccos(np.clip((ta@tc)/(na*nc), -1, 1))
        if b @ np.cross(ta, tc) < 0: ang = 2*np.pi - ang
        S += ang
    return S - (k-2)*np.pi

def medir(X, nome):
    coords, lam = embed(X)
    if coords is None: print(f"{nome}: embedding falhou"); return
    print(f"\n{nome}: lambda[1:4] = {lam}")
    print(f"  separação (l2-l1)/l1 = {(lam[1]-lam[0])/lam[0]:.4f}")
    res = {}
    for c in ciclos_curtos(X):
        seq = c + [c[0]]; d = []; ok = True
        for a, b in zip(seq, seq[1:]):
            t = coords[b] - coords[a]; nt = np.linalg.norm(t)
            if nt < 1e-14: ok = False; break
            d.append(t/nt)
        if not ok or len(d) < 3: continue
        O = angulo_solido(d)
        if O is not None: res.setdefault(len(c), []).append(O)
    print(f"  {'L':>3} {'n':>6} {'|Omega|':>9} {'desvio a 2pi':>13} "
          f"{'cos(O/2)':>10} {'frac neg':>9}")
    for L in sorted(res):
        a = np.array(res[L])
        if len(a) < 5: continue
        print(f"  {L:>3} {len(a):>6} {np.abs(a).mean():>9.4f} "
              f"{np.abs(np.abs(a)-2*np.pi).mean():>13.4f} "
              f"{np.cos(a/2).mean():>+10.4f} {(np.cos(a/2)<0).mean():>9.4f}")
    return res

medir(GC, "REDE REAL (componente gigante)")

# ---------------- modelo nulo
# O double_edge_swap preserva os graus exatamente, mas é demasiado lento
# a esta escala (horas). Usa-se o modelo de configuração, que gera a rede
# a partir da sequência de graus; as arestas múltiplas e os lacetes são
# removidos, pelo que a sequência fica aproximada. A perda é reportada.
print("\na construir o modelo nulo (modelo de configuração)...")
graus = [d for _, d in GC.degree()]
H = nx.configuration_model(graus, seed=3)
H = nx.Graph(H)
H.remove_edges_from(nx.selfloop_edges(H))
H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
print(f"  N={H.number_of_nodes():,} E={H.number_of_edges():,}  "
      f"(arestas perdidas: {1 - H.number_of_edges()/GC.number_of_edges():.1%})")
medir(H, "NULO (modelo de configuração)")
print(f"\ntotal {time.time()-t0:.0f}s")
