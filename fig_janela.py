import numpy as np, networkx as nx, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

OUT = "figuras"; LARG = 6.3; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 150, "savefig.bbox": "tight"})

def gera(N, rho, seed):
    rng = np.random.default_rng(seed)
    ar = [(0, 1)]; pares = {(0, 1)}; alvo = [0, 1]; n = 2
    pc = 1/(1+rho)
    while n < N:
        if rng.random() < pc or n < 3:
            u = alvo[rng.integers(len(alvo))]; v = n; n += 1
        else:
            for _ in range(50):
                u = alvo[rng.integers(len(alvo))]; v = alvo[rng.integers(len(alvo))]
                if u != v and (min(u,v), max(u,v)) not in pares: break
            else: continue
        p = (min(u,v), max(u,v))
        if p in pares: continue
        pares.add(p); ar.append((u, v)); alvo += [u, v]
    return ar

def obs(ar):
    nos = set(); [nos.update(e) for e in ar]
    idx = {v: i for i, v in enumerate(nos)}
    a = [(idx[u], idx[v]) for u, v in ar]
    N, E = len(nos), len(a)
    G = nx.Graph(); G.add_edges_from(a); c = nx.number_connected_components(G)
    tam = []; pdo = defaultdict(set)
    for u, v in a:
        d = pdo[u] ^ pdo[v]
        if d:
            m = min(d); nv = v if m in pdo[u] else u; tam[m] += 1; pdo[nv].add(m)
        else:
            m = len(tam); tam.append(2); pdo[u].add(m); pdo[v].add(m)
    rho = (E - N + c)/N
    Ctr = sum(len(s)-1 for s in pdo.values())/N
    return rho, Ctr/rho, max(tam)/N

fracs = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.13, 0.10]
sint = {s: [] for s in (3, 4)}
for s in sint:
    a = gera(60000, 1.0, s)
    for f in fracs:
        sint[s].append(obs(a[int(len(a)*(1-f)):]))

# pontos reais (janelas de 3, 5, 7, 9 dias)
real = [(0.0275, 2.81, 0.0843, "3 d"), (0.0340, 2.60, 0.1269, "5 d"),
        (0.0394, 2.46, 0.1535, "7 d"), (0.0431, 2.39, 0.1716, "9 d")]

fig, ax = plt.subplots(1, 2, figsize=(LARG, 2.6))
for s, pts in sint.items():
    pts = sorted(pts)
    r = [p[0] for p in pts]
    ax[0].plot(r, [p[1] for p in pts], "o-", ms=3, lw=0.9, color="0.55",
               label="rede sintética ($\\rho=1$), janelas" if s == 3 else None)
    ax[1].plot(r, [p[2] for p in pts], "o-", ms=3, lw=0.9, color="0.55",
               label="rede sintética ($\\rho=1$), janelas" if s == 3 else None)
rr = [p[0] for p in real]
ax[0].plot(rr, [p[1] for p in real], "s-", ms=4, lw=1.2, color="#c00000",
           label="rede de transações")
ax[1].plot(rr, [p[2] for p in real], "s-", ms=4, lw=1.2, color="#c00000",
           label="rede de transações")
for p in (real[0], real[-1]):
    ax[0].annotate(p[3], (p[0], p[1]), textcoords="offset points",
                   xytext=(4, 4), fontsize=7, color="#c00000")
    ax[1].annotate(p[3], (p[0], p[2]), textcoords="offset points",
                   xytext=(-16, 5) if p[3] == "3 d" else (5, -3),
                   fontsize=7, color="#c00000")
ax[0].axhline(1, ls="--", lw=0.8, color="0.3")
ax[0].text(0.35, 1.08, "$C_{tr}/N=\\rho$", fontsize=7, color="0.3")
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
vir = lambda v: (f"{v:g}").replace(".", ",")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
yt = [1, 1.5, 2, 3, 4]
ax[0].yaxis.set_major_locator(FixedLocator(yt))
ax[0].yaxis.set_major_formatter(FixedFormatter([vir(v) for v in yt]))
ax[0].yaxis.set_minor_locator(NullLocator())
ax[0].set_xlabel("$\\rho$"); ax[0].set_ylabel("$(C_{tr}/N)\\,/\\,\\rho$")
ax[0].set_title("(a) conservação dos ciclos", pad=4)
ax[1].set_xscale("log")
ax[1].set_xlabel("$\\rho$"); ax[1].set_ylabel("maior massa $/N$")
ax[1].set_title("(b) dimensão da maior massa", pad=4)
xt = [0.02, 0.05, 0.1, 0.2, 0.5, 1]
for a in ax:
    a.xaxis.set_major_locator(FixedLocator(xt))
    a.xaxis.set_major_formatter(FixedFormatter([vir(v) for v in xt]))
    a.xaxis.set_minor_locator(NullLocator())
    a.grid(alpha=0.25, lw=0.4)
ax[1].yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: vir(round(v, 2))))
ax[0].legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_janela_temporal.pdf")
print("ok")
