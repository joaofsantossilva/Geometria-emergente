"""
Geometria emergente por projeção espectral — pipeline
=====================================================

Substitui o notebook original. Diferenças em relação a ele, e porquê:

  1. COMPONENTE GIGANTE. O grafo original tinha 35 componentes, logo 35
     valores próprios nulos, e phi_1..phi_3 estavam TODOS no núcleo do
     Laplaciano. Não eram modos geométricos: eram indicadores de
     componente. O embedding degenerava em ~35 pontos.

  2. SEM PESOS. Com pesos em BTC, phi_1 localizava-se num único nó
     (IPR=0.985, curtose=156.5 ~ N). O modo mais baixo procura o corte
     mais barato e encontrava um nó ligado por uma transação de 2.2 BTC
     entre vizinhos com 1000+. Os "eixos geométricos" ordenavam
     carteiras por volume. Além disso, ponderar por volume importa
     informação que não é adjacência — contradiz a premissa da tese.

  3. SELF-LOOPS REMOVIDOS. 18 em 287 arestas. O Laplaciano ignora-os
     (somas de linha nulas), mas inflacionam a sequência de graus, que
     é o que define o modelo nulo.

  4. AMBIGUIDADE DE SINAL TRATADA. Vetores próprios definidos a menos
     de sinal. Sem fixar convenção, o embedding muda entre execuções.

  5. MODELOS NULOS. Sem comparador, "estrutura não trivial" não é
     afirmação verificável.

Uso:
    python pipeline_espectral.py --grafo g159_edgelist.txt
"""

import argparse
import numpy as np
import networkx as nx

SEED = 20260826


# ---------------------------------------------------------------------
# Preparação do grafo
# ---------------------------------------------------------------------
def preparar(G, verbose=True):
    """Limpa o grafo e devolve a componente gigante.

    O relatório desta função vai para a secção de metodologia da tese —
    todos os números que imprime devem ser reportados.
    """
    n0, e0 = G.number_of_nodes(), G.number_of_edges()

    loops = list(nx.selfloop_edges(G))
    G = G.copy()
    G.remove_edges_from(loops)
    G.remove_nodes_from([n for n in list(G.nodes()) if G.degree(n) == 0])

    ncomp = nx.number_connected_components(G)
    tam = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    if ncomp > 1:
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    N, E = G.number_of_nodes(), G.number_of_edges()
    if verbose:
        print(f"  original .......... N={n0}  E={e0}")
        print(f"  self-loops ........ {len(loops)} removidos")
        print(f"  componentes ....... {ncomp}  tamanhos={tam[:6]}")
        print(f"  componente gigante  N={N}  E={E}  beta_1={E-N+1}")
        deg = [d for _, d in G.degree()]
        print(f"  graus ............. min={min(deg)} max={max(deg)} "
              f"media={np.mean(deg):.2f}")
    return G


# ---------------------------------------------------------------------
# Embedding espectral
# ---------------------------------------------------------------------
def embedding(G, d=3, normalizado=False):
    """Projeção espectral em R^d a partir de adjacência pura.

    d=3 é POSTULADO do modelo. O espectro desta rede não apresenta
    nenhuma quebra que o justifique (razões ~1.9, 1.5, 1.3, 1.1 —
    decaimento suave). Não afirmar que a rede 'escolhe' 3 dimensões.
    """
    if normalizado:
        L = nx.normalized_laplacian_matrix(G, weight=None)
    else:
        L = nx.laplacian_matrix(G, weight=None)
    L = L.toarray().astype(float)

    ev, V = np.linalg.eigh(L)
    ev = np.clip(ev, 0.0, None)

    nulos = int(np.sum(ev < 1e-9))
    if nulos != 1:
        raise ValueError(
            f"{nulos} valores próprios nulos — grafo desconexo. "
            "phi_1..phi_d não são modos geométricos."
        )

    # Convenção de sinal: fixa a ambiguidade (Z_2)^d.
    # Sem isto, o embedding muda entre execuções.
    # NOTA TEÓRICA: esta liberdade discreta — e não SO(3) — é o grupo de
    # gauge que a construção espectral realmente deixa. As reflexões com
    # det=+1 formam V_4. Ver sub-pergunta 2.
    V = V.copy()
    for k in range(V.shape[1]):
        i = np.argmax(np.abs(V[:, k]))
        if V[i, k] < 0:
            V[:, k] *= -1

    return ev, V[:, 1:1+d], V


# ---------------------------------------------------------------------
# Observáveis
# ---------------------------------------------------------------------
def observaveis(G, ev, X, V, rng):
    """Observáveis do embedding.

    ATENÇÃO — o que NÃO medir: variâncias e covariâncias de X são
    fixadas pela ortonormalidade dos vetores próprios (C = I/(N-1)
    exatamente, para qualquer grafo conexo). Qualquer afirmação sobre
    'a forma da nuvem' a esse nível é vazia. Só os observáveis abaixo
    escapam a essa trivialidade.
    """
    N = G.number_of_nodes()
    d = X.shape[1]

    # Anisotropia do embedding escalado pelo espectro (commute-time).
    # A escala por 1/sqrt(lambda) é o que quebra a isotropia trivial.
    Y = X / np.sqrt(np.maximum(ev[1:1+d], 1e-12))
    w = np.linalg.eigvalsh(np.cov(Y.T))[::-1]
    anis = w / w[0]

    # Localização: nº efetivo de nós por modo (inverso do IPR).
    nef = [1.0/np.sum(V[:, k]**4) for k in range(1, 1+d)]

    # Curtose: 3 = gaussiana; ~N = modo concentrado num nó.
    kurt = [np.mean(X[:, k]**4)/np.mean(X[:, k]**2)**2 for k in range(d)]

    # Preservação da adjacência: arestas devem ficar mais curtas.
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    d_ar = np.mean([np.linalg.norm(X[idx[u]]-X[idx[v]]) for u, v in G.edges()])
    p = rng.integers(0, N, (4000, 2))
    d_na = np.mean([np.linalg.norm(X[a]-X[b]) for a, b in p if a != b])

    return {
        "anis2": anis[1], "anis3": anis[2],
        "nef1": nef[0], "nef2": nef[1], "nef3": nef[2],
        "kurt1": kurt[0],
        "gap21": ev[2]/ev[1],
        "razao_d": d_na/d_ar,
    }


def medir(G, rng, **kw):
    ev, X, V = embedding(G, **kw)
    return observaveis(G, ev, X, V, rng)


# ---------------------------------------------------------------------
# Modelos nulos
# ---------------------------------------------------------------------
def nulo_swap(G, rng, n_swap=20):
    """Double edge swap: preserva a sequência de graus EXATAMENTE.

    Preferível ao configuration model, que colapsa multi-arestas e
    remove self-loops, alterando ligeiramente os graus.
    """
    H = G.copy()
    try:
        nx.double_edge_swap(H, nswap=n_swap*H.number_of_edges(),
                            max_tries=500*H.number_of_edges(),
                            seed=int(rng.integers(1e9)))
    except (nx.NetworkXError, nx.NetworkXAlgorithmError):
        return None
    if not nx.is_connected(H):
        H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
    return H


def nulo_config(deg, rng):
    """Configuration model. Reportar o desvio de graus que introduz."""
    H = nx.Graph(nx.configuration_model(deg, seed=int(rng.integers(1e9))))
    H.remove_edges_from(list(nx.selfloop_edges(H)))
    if H.number_of_nodes() == 0 or not nx.is_connected(H):
        if H.number_of_nodes() == 0:
            return None
        H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
    return H


def ensemble(gerador, n, rng, **kw):
    out = []
    for _ in range(n):
        H = gerador()
        if H is None or H.number_of_nodes() < 10:
            continue
        try:
            out.append(medir(H, rng, **kw))
        except ValueError:
            continue
    return out


def comparar(real, amostras, titulo):
    print(f"\n{titulo}  ({len(amostras)} realizações)")
    print(f"  {'observável':10s} {'real':>9s} {'nulo (media±dp)':>21s} {'z':>7s}")
    print("  " + "-"*50)
    linhas = {}
    for k in real:
        v = np.array([a[k] for a in amostras])
        z = (real[k]-v.mean())/v.std() if v.std() > 1e-12 else np.nan
        linhas[k] = z
        marca = " *" if abs(z) > 3 else ""
        print(f"  {k:10s} {real[k]:9.3f} {v.mean():12.3f} ±{v.std():6.3f} "
              f"{z:7.1f}{marca}")
    return linhas


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grafo", required=True, help="ficheiro edgelist")
    ap.add_argument("--n-nulo", type=int, default=200)
    ap.add_argument("--normalizado", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)

    print("="*56)
    print("REDE EMPÍRICA")
    print("="*56)
    G = preparar(nx.read_edgelist(args.grafo, nodetype=int))
    real = medir(G, rng, normalizado=args.normalizado)
    print("\n  observáveis:")
    for k, v in real.items():
        print(f"    {k:10s} {v:8.3f}")

    deg = [d for _, d in G.degree()]
    N = G.number_of_nodes()

    print("\n" + "="*56)
    print("MODELOS NULOS")
    print("="*56)

    comparar(real,
             ensemble(lambda: nulo_swap(G, rng), args.n_nulo, rng,
                      normalizado=args.normalizado),
             "Double edge swap (graus preservados exatamente)")

    comparar(real,
             ensemble(lambda: nulo_config(deg, rng), args.n_nulo, rng,
                      normalizado=args.normalizado),
             "Configuration model (graus aproximados)")

    print("\n" + "="*56)
    print("REFERÊNCIA: BA(m=1)")
    print("="*56)
    ba = nx.barabasi_albert_graph(N, 1, seed=SEED)
    ref = medir(ba, rng, normalizado=args.normalizado)
    print(f"  BA(m=1) N={N}  beta_1={ba.number_of_edges()-N+1} (árvore)")
    print(f"  {'observável':10s} {'BTC':>9s} {'BA(m=1)':>9s}")
    print("  " + "-"*30)
    for k in real:
        print(f"  {k:10s} {real[k]:9.3f} {ref[k]:9.3f}")

    # Nulo próprio da BA: árvore aleatória com a mesma sequência de graus.
    # O configuration model NÃO serve para a BA — gera ciclos, e a
    # comparação atribuiria à "geometria" o que é diferença topológica.
    print("\n  Nulo da BA: árvore aleatória (mesma sequência de graus)")
    amostras = []
    for _ in range(args.n_nulo):
        T = nx.random_labeled_tree(N, seed=int(rng.integers(1e9))) \
            if hasattr(nx, "random_labeled_tree") \
            else nx.random_tree(N, seed=int(rng.integers(1e9)))
        try:
            amostras.append(medir(T, rng, normalizado=args.normalizado))
        except ValueError:
            pass
    comparar(ref, amostras, "  BA(m=1) vs. árvore aleatória")


if __name__ == "__main__":
    main()
