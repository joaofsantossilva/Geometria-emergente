# =====================================================================
# Holonomia nos ciclos: rede real contra modelo nulo, à mesma escala
#
#  O embedding da componente gigante (851 864 nós) é caro e, no nulo,
#  o shift-invert não converge em tempo útil. Reduz-se a escala: extrai-se
#  uma sub-rede conexa da rede real por pesquisa em largura a partir de
#  um nó, com N_ALVO nós, e constrói-se o nulo com a MESMA sequência de
#  graus dessa sub-rede. As duas redes ficam assim comparáveis em
#  dimensão e em densidade, e o embedding é rápido.
#
#  Nos triângulos o resultado é exato em qualquer embedding, pelo que a
#  redução de escala não o afeta; serve para os ciclos mais longos.
# =====================================================================
import numpy as np, pandas as pd, networkx as nx, time
from collections import deque
from scipy.sparse.linalg import eigsh

FICHEIRO = "bq_BTC_results.csv"
N_ALVO, LMAX, MAXC = 60000, 8, 4000
t0 = time.time()

df = pd.read_csv(FICHEIRO, usecols=["src", "dst"])
df = df.dropna(subset=["src", "dst"]); df = df[df.src != df.dst]
cod = pd.factorize(pd.concat([df.src, df.dst]), sort=False)[0]
n = len(df)
G = nx.Graph(); G.add_edges_from(zip(cod[:n], cod[n:]))
GC = G.subgraph(max(nx.connected_components(G), key=len)).copy()
print(f"componente gigante: N={GC.number_of_nodes():,} "
      f"E={GC.number_of_edges():,}  ({time.time()-t0:.0f}s)")

# ---- sub-rede conexa por BFS a partir do nó de maior grau
raiz = max(GC.degree, key=lambda t: t[1])[0]
vis = {raiz}; fila = deque([raiz])
while fila and len(vis) < N_ALVO:
    x = fila.popleft()
    for y in GC[x]:
        if y not in vis:
            vis.add(y); fila.append(y)
            if len(vis) >= N_ALVO: break
S = GC.subgraph(vis).copy()
S = S.subgraph(max(nx.connected_components(S), key=len)).copy()
b1 = S.number_of_edges() - S.number_of_nodes() + 1
print(f"sub-rede: N={S.number_of_nodes():,} E={S.number_of_edges():,} "
      f"beta_1={b1:,}  rho={b1/S.number_of_nodes():.4f}")

def embed(X, d=3):
    L = nx.laplacian_matrix(X).astype(float).tocsc(); N = X.number_of_nodes()
    w = None
    for sig in (-1e-4, -1e-2, -1e-1):      # sigma maior: converge melhor
        try:
            w, V = eigsh(L, k=d+3, sigma=sig, which="LM", maxiter=5000)
            break
        except Exception:
            w = None
    if w is None: return None, None
    o = np.argsort(w); w, V = np.clip(w[o], 0, None), V[:, o]
    nz = [j for j in range(len(w)) if w[j] > 1e-12
          and abs(V[:, j].sum())/np.sqrt(N) < 1e-6]
    if len(nz) < d: return None, None
    Y = V[:, nz[:d]]
    # Normalização por eixo: cada modo é dividido pelo seu desvio-padrão.
    # Sem isto, um embedding quase degenerado colapsa num plano e todos
    # os ciclos saem coplanares, dando Omega = 2pi por construção e não
    # por propriedade da rede. A normalização põe os três eixos na mesma
    # escala e torna as duas redes comparáveis.
    sd = Y.std(axis=0); sd[sd < 1e-15] = 1.0
    Y = Y / sd
    return {x: Y[i] for i, x in enumerate(X.nodes())}, w[nz[:d]]

def ciclos_curtos(X, Lmax=LMAX, maxn=MAXC, seed=0):
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
    k = len(dirs); S = 0.0
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
    t = time.time(); coords, lam = embed(X)
    if coords is None: print(f"{nome}: embedding falhou"); return
    print(f"\n{nome}  (embedding em {time.time()-t:.0f}s)")
    print(f"  lambda[1:4] = {lam}   separação {(lam[1]-lam[0])/lam[0]:.4f}")
    print(f"  (eixos normalizados pelo desvio-padrão de cada modo)")
    res = {}
    for c in ciclos_curtos(X):
        seq = c + [c[0]]; d = []; ok = True
        for a, b in zip(seq, seq[1:]):
            t2 = coords[b] - coords[a]; nt = np.linalg.norm(t2)
            if nt < 1e-14: ok = False; break
            d.append(t2/nt)
        if not ok or len(d) < 3: continue
        O = angulo_solido(d)
        if O is not None: res.setdefault(len(c), []).append(O)
    print(f"  {'L':>3} {'n':>6} {'desvio a 2pi':>13} {'cos(O/2)':>10} {'frac neg':>9}")
    for L in sorted(res):
        a = np.array(res[L])
        if len(a) < 5: continue
        print(f"  {L:>3} {len(a):>6} {np.abs(np.abs(a)-2*np.pi).mean():>13.4f} "
              f"{np.cos(a/2).mean():>+10.4f} {(np.cos(a/2)<0).mean():>9.4f}")

medir(S, "SUB-REDE REAL")

graus = [d for _, d in S.degree()]
H = nx.configuration_model(graus, seed=3)
H = nx.Graph(H); H.remove_edges_from(nx.selfloop_edges(H))
H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
b1H = H.number_of_edges() - H.number_of_nodes() + 1
print(f"\nnulo: N={H.number_of_nodes():,} E={H.number_of_edges():,} "
      f"rho={b1H/H.number_of_nodes():.4f}  "
      f"(arestas perdidas: {1-H.number_of_edges()/S.number_of_edges():.1%})")
medir(H, "NULO (mesma sequência de graus)")
print(f"\ntotal {time.time()-t0:.0f}s")
