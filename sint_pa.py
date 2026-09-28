import numpy as np
from collections import defaultdict
from teste_pa import gera, massas

def obs_rede(ar):
    g = defaultdict(int)
    for u, v in ar: g[u] += 1; g[v] += 1
    k = np.array(list(g.values()))
    return (k % 2 == 1).mean(), (k == 1).mean()          # ímpar, folhas

def obs_massa(arestas):
    g = defaultdict(int)
    for u, v in arestas: g[u] += 1; g[v] += 1
    k = np.array(list(g.values()))
    casca = sum(1 for u, v in arestas if g[u] == 1 or g[v] == 1)
    return (k == 1).mean(), casca / len(arestas), casca

res = {}
for pref in (True, False):
    for rho in (0.0, 0.043, 0.25, 0.5, 1.0, 2.0):
        linhas = []
        for s in range(5):
            ar = gera(20000, rho, pref, seed=100 + s)
            impar, folhas = obs_rede(ar)
            pat = massas(ar)
            bb = max(range(len(pat)), key=lambda i: len(pat[i]))
            p1_bb, casca_bb, _ = obs_massa(pat[bb])
            sat = [obs_massa(pat[i])[2] for i in range(len(pat)) if i != bb]
            ferm = np.mean([c % 2 for c in sat]) if sat else np.nan
            linhas.append((impar, folhas, p1_bb, casca_bb, ferm))
        res[(pref, rho)] = np.array(linhas)

print("5 realizações, N=20000; média ± desvio-padrão\n")
print(f"{'entrada':12s} {'rho':>6s} | {'ímpar rede':>13s} {'folhas rede':>13s} | "
      f"{'casca backbone':>15s} | {'fermiónica sat.':>15s}")
for (pref, rho), a in res.items():
    m, d = a.mean(0), a.std(0)
    nome = "preferencial" if pref else "uniforme"
    f = lambda i: f"{m[i]:.3f}±{d[i]:.3f}"
    print(f"{nome:12s} {rho:6.3f} | {f(0):>13s} {f(1):>13s} | {f(3):>15s} | "
          f"{'—' if np.isnan(m[4]) else f(4):>15s}")
print("\nreferências analíticas BA(m=1): ímpar = 4ln2-2 = 0,7726 ; folhas = P(1) = 2/3 = 0,6667")
