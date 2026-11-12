"""
Centralidade de intermediação (betweenness) - algoritmo de Brandes (2001), implementado do zero.

Para grafos ponderados, o Brandes roda um Dijkstra a partir de cada vértice s,
registrando para cada vértice v:
  sigma[v] : número de caminhos mínimos de s até v
  P[v]     : predecessores de v nos caminhos mínimos
  S        : pilha com os vértices na ordem em que foram fixados (distância não decrescente)

Depois, percorre S de trás para frente acumulando a "dependência" delta[v]:
  delta[v] = soma_{w : v em P[w]} (sigma[v] / sigma[w]) * (1 + delta[w])
e soma delta[v] à centralidade de v (para v != s).

Complexidade: O(V*E + V^2 log V) com fila de prioridade (heap).

Representação: lista de adjacência  adj[u] = [(v, peso_distancia), ...]
No nosso modelo, a distância de uma aresta é 1 / (nº de transferências entre os clubes):
quanto mais transferências, "mais curto" é o caminho entre os dois clubes.
"""
import heapq
from itertools import count


def brandes(adj: dict, normalizar: bool = True, origens=None, alvos=None) -> dict:
    """
    origens/alvos (opcionais): restringe a contagem aos caminhos mínimos que SAEM de um
    vértice em `origens` e CHEGAM a um vértice em `alvos` ("intermediação dirigida").
    É a métrica central do trabalho: com origens = clubes sul-americanos e alvos = clubes
    das 5 grandes ligas, mede o quanto cada clube está no caminho entre os dois grupos.
    Nesse modo não há normalização (retorna o valor bruto de pares intermediados).
    """
    vertices = list(adj.keys())
    for u in list(adj):
        for v, _ in adj[u]:
            if v not in adj:
                vertices.append(v)
                adj[v] = []
    cb = {v: 0.0 for v in vertices}
    subconjunto = origens is not None or alvos is not None
    if origens is None:
        fontes = vertices
    else:
        conj_origens = set(origens)
        fontes = [v for v in vertices if v in conj_origens]
    eh_alvo = (lambda w: True) if alvos is None else set(alvos).__contains__

    for s in fontes:
        # ---------- fase 1: Dijkstra contando caminhos mínimos ----------
        S = []
        P = {v: [] for v in vertices}
        sigma = dict.fromkeys(vertices, 0.0)
        sigma[s] = 1.0
        dist = {}                      # distâncias definitivas
        visto = {s: 0.0}               # melhor distância conhecida
        desempate = count()            # evita comparar vértices no heap
        fila = [(0.0, next(desempate), s, s)]
        while fila:
            d, _, pred, v = heapq.heappop(fila)
            if v in dist:
                continue
            sigma[v] += sigma[pred] if pred != v else 0.0
            S.append(v)
            dist[v] = d
            for w, peso in adj[v]:
                nd = d + peso
                if w not in dist and (w not in visto or nd < visto[w]):
                    visto[w] = nd
                    heapq.heappush(fila, (nd, next(desempate), v, w))
                    sigma[w] = 0.0
                    P[w] = [v]
                elif nd == visto.get(w):   # outro caminho mínimo de mesmo comprimento
                    sigma[w] += sigma[v]
                    P[w].append(v)

        # ---------- fase 2: acumulação das dependências ----------
        delta = dict.fromkeys(S, 0.0)
        while S:
            w = S.pop()
            # cada caminho mínimo que TERMINA em w conta 1 (se w for alvo) + o que passa por w
            coef = ((1.0 if (eh_alvo(w) and w != s) else 0.0) + delta[w]) / sigma[w]
            for v in P[w]:
                delta[v] += sigma[v] * coef
            if w != s:
                cb[w] += delta[w]

    if normalizar and not subconjunto:
        n = len(vertices)
        if n > 2:
            escala = 1.0 / ((n - 1) * (n - 2))   # grafo dirigido
            for v in cb:
                cb[v] *= escala
    return cb


def adjacencia_de_arestas(arestas) -> dict:
    """arestas: iterável de (origem, destino, n_transferencias)."""
    adj = {}
    for u, v, n in arestas:
        adj.setdefault(u, []).append((v, 1.0 / n))
        adj.setdefault(v, [])
    return adj
