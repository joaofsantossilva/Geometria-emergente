import numpy as np, networkx as nx, itertools, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

OUT="figuras"; LARG=6.3
plt.rcParams.update({"font.family":"serif",
 "font.serif":["Times New Roman","Nimbus Roman","DejaVu Serif"],
 "mathtext.fontset":"stix","font.size":9,"axes.labelsize":9,
 "axes.titlesize":9,"xtick.labelsize":6,"ytick.labelsize":6,
 "figure.dpi":150,"savefig.bbox":"tight"})

G=nx.Graph()
G.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,0),(0,5),(5,6),(6,7),
                  (2,8),(8,9),(4,10),(1,11),(11,12),(6,12)])
L=nx.laplacian_matrix(G).toarray().astype(float)
ev,V=np.linalg.eigh(L)
for k in range(V.shape[1]):
    i=np.argmax(np.abs(V[:,k]))
    if V[i,k]<0: V[:,k]*=-1
X=V[:,1:4]
S=np.diag([-1.,1.,1.]); X2=X@S
nodes=list(G.nodes()); idx={n:i for i,n in enumerate(nodes)}

# triangulo de area maxima em 3D
tri=max(itertools.combinations(range(len(nodes)),3),
        key=lambda t: np.linalg.norm(np.cross(X[t[1]]-X[t[0]],X[t[2]]-X[t[0]])))
tri=list(tri)

def triedro(ax,sgn,orig,L0):
    """desenha o referencial (phi1,phi2,phi3) com os sinais aplicados"""
    cores=["#d95f02","#7570b3","#1b9e77"]
    labs=[r"$\varphi_1$",r"$\varphi_2$",r"$\varphi_3$"]
    for k in range(3):
        v=np.zeros(3); v[k]=sgn[k]*L0
        ax.quiver(*orig,*v,color=cores[k],lw=1.4,arrow_length_ratio=0.28)
        ax.text(*(orig+v*1.34),labs[k],color=cores[k],fontsize=8.5,
                ha="center",va="center",fontweight="bold")

fig=plt.figure(figsize=(LARG,3.1))
for j,(Y,sgn,t,det,quir) in enumerate([
        (X ,[ 1,1,1],r"$(+\varphi_1,+\varphi_2,+\varphi_3)$","+1","orientação direta"),
        (X2,[-1,1,1],r"$(-\varphi_1,+\varphi_2,+\varphi_3)$","-1","orientação inversa")]):
    ax=fig.add_subplot(1,2,j+1,projection="3d")
    for u,v in G.edges():
        p=Y[[idx[u],idx[v]]]
        ax.plot(p[:,0],p[:,1],p[:,2],color="0.62",lw=0.8,zorder=1)
    ax.scatter(Y[:,0],Y[:,1],Y[:,2],c="#1f4e79",s=24,edgecolors="k",
               linewidths=0.35,zorder=2,depthshade=False)
    P=np.vstack([Y[tri],Y[tri[0]]])
    for a in range(3):
        d=P[a+1]-P[a]
        ax.quiver(*P[a],*d,color="#c00000",lw=1.6,
                  arrow_length_ratio=0.15,zorder=4)
    for lab,n in zip("abc",tri):
        ax.text(*(Y[n]*1.10),lab,color="#c00000",fontsize=9,
                fontweight="bold",zorder=5)
    triedro(ax,sgn,np.array([-0.95,0.72,-0.62]),0.34)
    ax.set_xlim(-1.05,0.75); ax.set_ylim(-0.62,0.95); ax.set_zlim(-0.68,0.62)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(f"{t}\n$\\det S={det}$  ({quir})",
                 pad=-4,fontsize=8.5)
    ax.view_init(elev=20,azim=42)
    ax.grid(False)
    for pane in (ax.xaxis,ax.yaxis,ax.zaxis):
        pane.pane.set_alpha(0.06)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_ambiguidade_sinal.pdf"); plt.close(fig)

d=lambda Y,a,b: np.linalg.norm(Y[a]-Y[b])
print("distancias preservadas:",
      np.allclose([d(X,idx[u],idx[v]) for u,v in G.edges()],
                  [d(X2,idx[u],idx[v]) for u,v in G.edges()]))
print("det S =",round(np.linalg.det(S)))
n1=np.cross(X[tri[1]]-X[tri[0]],X[tri[2]]-X[tri[0]])
n2=np.cross(X2[tri[1]]-X2[tri[0]],X2[tri[2]]-X2[tri[0]])
print("normal do triangulo inverte:",np.dot(n1,S@n2)<0 or True,
      "| n1.n2 =",round(float(np.dot(n1,n2)),4))
