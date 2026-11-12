"""
Etapa 2 - Modelagem do grafo e análises.

Grafo G = (V, E) dirigido e ponderado:
  V : clubes das 32 ligas acompanhadas pela base (796)
  E : (u -> v) se houve ao menos uma transferência de u para v entre 2010 e 2026
      peso  w(u,v) = nº de transferências de u para v
      distância d(u,v) = 1 / w(u,v)   (usada nos caminhos mínimos)

Análises (cada uma responde uma parte da pergunta de pesquisa):
  A1. Intermediação global (Brandes, implementação própria) + validação contra o NetworkX
  A2. Intermediação dirigida América do Sul -> Top 5 (clubes-ponte)          [pergunta principal]
  A3. Caminhos reais de carreira (o que de fato aconteceu com os jogadores)
  A4. Evolução temporal dos clubes-ponte
  A5. PageRank ponderado (atração de jogadores)
  A6. Comunidades (Louvain) e comparação com a divisão por país
  A7. Saldo de jogadores e de dinheiro por país
  A8. Escalabilidade do Brandes (tempo x tamanho do grafo)

Uso:  python3 src/analise.py      (rodar antes: python3 src/preprocess.py)
Saídas em results/tabelas, results/figuras e results/resumo.txt
"""
import random
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brandes import adjacencia_de_arestas, brandes  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
TRATADO = RAIZ / "data" / "processed"
BRUTO = RAIZ / "data" / "raw"
FIG = RAIZ / "results" / "figuras"
TAB = RAIZ / "results" / "tabelas"

SEMENTE = 42
TOP5 = {"England", "Spain", "Italy", "Germany", "France"}
AMERICA_SUL = {"Brazil", "Argentina", "Colombia"}
PAIS_PT = {"England": "Inglaterra", "Spain": "Espanha", "Italy": "Itália", "Germany": "Alemanha",
           "France": "França", "Brazil": "Brasil", "Argentina": "Argentina", "Colombia": "Colômbia", "Portugal": "Portugal",
           "Netherlands": "Holanda", "Belgium": "Bélgica", "Türkiye": "Turquia", "Ukraine": "Ucrânia",
           "Russia": "Rússia", "Greece": "Grécia", "Mexico": "México", "United States": "EUA",
           "Switzerland": "Suíça", "Austria": "Áustria", "Japan": "Japão", "Saudi Arabia": "Arábia Saudita",
           "Scotland": "Escócia", "Denmark": "Dinamarca", "Croatia": "Croácia", "Serbia": "Sérvia",
           "Poland": "Polônia", "Czech Republic": "Rep. Tcheca", "Romania": "Romênia", "Sweden": "Suécia",
           "Norway": "Noruega", "Korea, South": "Coreia do Sul", "Australia": "Austrália"}

# paleta categorical (dataviz reference palette, slots 1-3) + tintas de texto
COR = {"Top 5 Europa": "#2a78d6", "América do Sul": "#eb6834", "Outros": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.size": 7, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

resumo = []


def log(txt=""):
    print(txt)
    resumo.append(txt)


def salvar(fig, nome):
    fig.savefig(FIG / f"{nome}.pdf")
    fig.savefig(FIG / f"{nome}.png")
    plt.close(fig)


NOME_CURTO = {"Olympiakos Syndesmos Filathlon Peiraios": "Olympiacos", "Olympiakos Syndesmos Filathlon": "Olympiacos",
              "AO FK Zenit Sankt-Peterburg": "Zenit", "FC Shakhtar Donetsk": "Shakhtar Donetsk",
              "Associazione Sportiva Roma": "Roma"}


def curto(nome):
    for longo, c in NOME_CURTO.items():
        if nome.startswith(longo):
            return c
    return nome


def pt(pais):
    return PAIS_PT.get(pais, pais)


def carregar():
    clubes = pd.read_csv(TRATADO / "clubes.csv")
    arestas = pd.read_csv(TRATADO / "arestas.csv")
    transf = pd.read_csv(TRATADO / "transferencias.csv", parse_dates=["data"])
    return clubes, arestas, transf


def grafo_nx(arestas):
    G = nx.DiGraph()
    for u, v, n, val in arestas[["origem", "destino", "n_transferencias", "valor_total_eur"]].itertuples(
            index=False, name=None):
        G.add_edge(u, v, peso=n, dist=1.0 / n, valor=val)
    return G


# ---------------------------------------------------------------- A1
def a1_intermediacao_global(arestas, clubes, G):
    log("== A1. Intermediação global (Brandes) ==")
    E = list(arestas[["origem", "destino", "n_transferencias"]].itertuples(index=False, name=None))
    t0 = time.perf_counter()
    bc = brandes(adjacencia_de_arestas(E))
    t_nosso = time.perf_counter() - t0
    t0 = time.perf_counter()
    bc_nx = nx.betweenness_centrality(G, weight="dist")
    t_nx = time.perf_counter() - t0
    dif = max(abs(bc[v] - bc_nx[v]) for v in bc_nx)
    log(f"V={G.number_of_nodes()}  E={G.number_of_edges()}  "
        f"componentes fortemente conexas={nx.number_strongly_connected_components(G)}  "
        f"maior CFC={len(max(nx.strongly_connected_components(G), key=len))}")
    log(f"Validação: maior diferença vs NetworkX = {dif:.2e}  "
        f"(tempo nosso={t_nosso:.2f}s, NetworkX={t_nx:.2f}s)")

    grau = dict(G.degree(weight="peso"))
    df = clubes.set_index("clube_id").assign(
        intermediacao=pd.Series(bc), grau_ponderado=pd.Series(grau),
        grau_entrada=pd.Series(dict(G.in_degree(weight="peso"))),
        grau_saida=pd.Series(dict(G.out_degree(weight="peso"))),
    ).fillna(0).sort_values("intermediacao", ascending=False)
    rho, _ = spearmanr(df["intermediacao"], df["grau_ponderado"])
    log(f"Correlação de Spearman intermediação x grau ponderado: {rho:.3f}")
    df["rank_intermediacao"] = df["intermediacao"].rank(ascending=False).astype(int)
    df["rank_grau"] = df["grau_ponderado"].rank(ascending=False, method="min").astype(int)
    df.to_csv(TAB / "a1_intermediacao_global.csv")
    log("Top 10:")
    for cid, r in df.head(10).iterrows():
        log(f"  {curto(r['nome'])[:28]:28} {pt(r['pais']):12} BC={r['intermediacao']:.4f} "
            f"(rank grau {r['rank_grau']})")

    # figura: grau x intermediação
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for reg, cor in COR.items():
        d = df[df["regiao"] == reg]
        ax.scatter(d["grau_ponderado"], d["intermediacao"], s=12, color=cor, alpha=0.75,
                   edgecolor="white", linewidth=0.4, label=reg)
    deslocamento = {0: (-4, 4, "right"), 1: (5, -2, "left"), 2: (5, 2, "left"), 3: (5, 2, "left"),
                    4: (5, 2, "left")}
    for i, (cid, r) in enumerate(df.head(5).iterrows()):
        dx, dy, ha = deslocamento[i]
        ax.annotate(curto(r["nome"]), (r["grau_ponderado"], r["intermediacao"]), fontsize=6,
                    color=INK2, xytext=(dx, dy), textcoords="offset points", ha=ha)
    ax.set_xlabel("Grau ponderado (nº de transferências de entrada + saída)")
    ax.set_ylabel("Centralidade de intermediação")
    ax.legend(frameon=False, loc="upper left")
    salvar(fig, "a1_grau_vs_intermediacao")
    return df, t_nosso


# ---------------------------------------------------------------- A2
def a2_clubes_ponte(arestas, clubes, rotulo="2010-2026", salvar_saidas=True):
    E = list(arestas[["origem", "destino", "n_transferencias"]].itertuples(index=False, name=None))
    origens = clubes.loc[clubes["regiao"] == "América do Sul", "clube_id"].tolist()
    alvos = clubes.loc[clubes["regiao"] == "Top 5 Europa", "clube_id"].tolist()
    bc = brandes(adjacencia_de_arestas(E), origens=origens, alvos=alvos)
    df = clubes.set_index("clube_id").assign(intermediacao_sa_top5=pd.Series(bc)).fillna(0)
    # um clube-ponte é um clube que não é nem origem nem destino final
    pontes = df[df["regiao"] == "Outros"].sort_values("intermediacao_sa_top5", ascending=False)
    total = df["intermediacao_sa_top5"].sum()
    if not salvar_saidas:
        return pontes, total

    log("\n== A2. Clubes-ponte: intermediação dirigida América do Sul -> Top 5 ==")
    log(f"Origens (clubes BRA+ARG+COL)={len(origens)}  Alvos (clubes Top 5)={len(alvos)}")
    ref = nx.betweenness_centrality_subset(grafo_nx(arestas), origens, alvos, weight="dist", normalized=False)
    log(f"Validação: maior diferença vs NetworkX (subset) = {max(abs(bc[v] - ref[v]) for v in ref):.2e}")
    df.sort_values("intermediacao_sa_top5", ascending=False).to_csv(TAB / "a2_intermediacao_sa_top5.csv")
    por_pais = (pontes.groupby("pais")["intermediacao_sa_top5"].sum() / pontes["intermediacao_sa_top5"].sum()
                ).sort_values(ascending=False)
    por_pais.rename("participacao").to_csv(TAB / "a2_pontes_por_pais.csv")
    log("Top 10 clubes-ponte (fora da América do Sul e do Top 5):")
    for cid, r in pontes.head(10).iterrows():
        log(f"  {curto(r['nome'])[:28]:28} {pt(r['pais']):12} {r['intermediacao_sa_top5']:8.1f}")
    log("Participação de cada país na intermediação (entre os clubes-ponte):")
    for p, v in por_pais.head(6).items():
        log(f"  {pt(p):12} {v:6.1%}")

    top = pontes.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    ax.barh([curto(n) for n in top["nome"]], top["intermediacao_sa_top5"], color=COR["Outros"],
            height=0.65)
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r["intermediacao_sa_top5"], i, f"  {pt(r['pais'])}", va="center", fontsize=6, color=MUTED)
    ax.set_xlabel("Pares (clube sul-americano, clube Top 5) intermediados")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, top["intermediacao_sa_top5"].max() * 1.22)
    salvar(fig, "a2_clubes_ponte")
    return pontes, total


# ---------------------------------------------------------------- A3
def a3_caminhos_reais(clubes):
    """Usa as transferências BRUTAS para não esconder escalas em clubes fora da base."""
    log("\n== A3. Caminhos reais de carreira (América do Sul -> Top 5) ==")
    t = pd.read_csv(BRUTO / "transfers.csv")
    t["data"] = pd.to_datetime(t["transfer_date"], errors="coerce")
    t = t[t["data"].dt.year.between(2010, 2026)].sort_values(["player_id", "data"])
    pais = dict(zip(clubes["clube_id"], clubes["pais"]))
    nome = dict(zip(clubes["clube_id"], clubes["nome"]))
    t["po"] = t["from_club_id"].map(pais)
    t["pd"] = t["to_club_id"].map(pais)

    direto = via = nunca = 0
    escala_pais, escala_clube, saltos = Counter(), Counter(), []
    for _, g in t.groupby("player_id"):
        g = g.reset_index(drop=True)
        saida = g.index[g["po"].isin(AMERICA_SUL) & ~g["pd"].isin(AMERICA_SUL)]
        if len(saida) == 0:
            continue
        resto = g.loc[saida[0]:]
        chegada = resto.index[resto["pd"].isin(TOP5)]
        if len(chegada) == 0:
            nunca += 1
            continue
        meio = g.loc[saida[0]:chegada[0] - 1]
        meio = meio[~meio["pd"].isin(AMERICA_SUL)]
        saltos.append(len(meio) + 1)
        if meio.empty:
            direto += 1
            continue
        via += 1
        primeiro = meio.iloc[0]
        escala_pais[pt(primeiro["pd"]) if isinstance(primeiro["pd"], str) else "Fora da base"] += 1
        if primeiro["to_club_id"] in nome:
            escala_clube[curto(nome[primeiro["to_club_id"]])] += 1

    chegaram = direto + via
    log(f"Jogadores que saíram de clubes BRA/ARG/COL para o exterior: {direto + via + nunca}")
    log(f"  chegaram ao Top 5: {chegaram} ({chegaram / (direto + via + nunca):.1%})")
    log(f"    direto: {direto} ({direto / chegaram:.1%})   com escala: {via} ({via / chegaram:.1%})")
    log(f"  média de transferências até o Top 5: {np.mean(saltos):.2f}")
    log("País da primeira escala:")
    for p, n in escala_pais.most_common(8):
        log(f"  {p:14} {n:4} ({n / via:.1%})")
    log("Clubes mais usados como primeira escala:")
    for c, n in escala_clube.most_common(8):
        log(f"  {c[:28]:28} {n}")
    pd.DataFrame(escala_pais.most_common(), columns=["pais", "jogadores"]).to_csv(
        TAB / "a3_escala_por_pais.csv", index=False)
    pd.DataFrame(escala_clube.most_common(), columns=["clube", "jogadores"]).to_csv(
        TAB / "a3_escala_por_clube.csv", index=False)

    dados = escala_pais.most_common(8)[::-1]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    cores = [MUTED if p == "Fora da base" else COR["Outros"] for p, _ in dados]
    ax.barh([p for p, _ in dados], [n for _, n in dados], color=cores, height=0.65)
    for i, (_, n) in enumerate(dados):
        ax.text(n, i, f" {n}", va="center", fontsize=6, color=INK2)
    ax.set_xlabel("Jogadores (primeira escala antes de chegar ao Top 5)")
    ax.grid(axis="y", visible=False)
    salvar(fig, "a3_escalas_por_pais")
    return escala_clube


# ---------------------------------------------------------------- A4
def a4_temporal(transf, clubes):
    log("\n== A4. Evolução temporal dos clubes-ponte ==")
    periodos = [(2010, 2014), (2015, 2019), (2020, 2026)]
    linhas = []
    for a, b in periodos:
        sub = transf[transf["data"].dt.year.between(a, b)]
        ar = sub.groupby(["origem", "destino"]).size().rename("n_transferencias").reset_index()
        pontes, _ = a2_clubes_ponte(ar, clubes, salvar_saidas=False)
        share = pontes.groupby("pais")["intermediacao_sa_top5"].sum()
        share = share / share.sum()
        log(f"{a}-{b}: top 3 = " + ", ".join(f"{n} ({pt(p)})" for n, p in
                                              zip(pontes.head(3)["nome"], pontes.head(3)["pais"])))
        log(f"         participação: " + ", ".join(f"{pt(p)} {v:.0%}" for p, v in
                                                   share.sort_values(ascending=False).head(4).items()))
        for p, v in share.items():
            linhas.append({"periodo": f"{a}-{b}", "pais": p, "participacao": v})
    pd.DataFrame(linhas).to_csv(TAB / "a4_temporal.csv", index=False)


# ---------------------------------------------------------------- A5
def a5_pagerank(G, clubes):
    log("\n== A5. PageRank ponderado (atração de jogadores) ==")
    pr = nx.pagerank(G, weight="peso", alpha=0.85)
    df = clubes.set_index("clube_id").assign(pagerank=pd.Series(pr)).sort_values("pagerank", ascending=False)
    df.to_csv(TAB / "a5_pagerank.csv")
    for _, r in df.head(10).iterrows():
        log(f"  {curto(r['nome'])[:28]:28} {pt(r['pais']):12} {r['pagerank']:.4f}")


# ---------------------------------------------------------------- A6
def a6_comunidades(G, clubes):
    log("\n== A6. Comunidades (Louvain) ==")
    U = nx.Graph()
    for u, v, d in G.edges(data=True):
        w = U[u][v]["peso"] + d["peso"] if U.has_edge(u, v) else d["peso"]
        U.add_edge(u, v, peso=w)
    com = nx.community.louvain_communities(U, weight="peso", seed=SEMENTE)
    q_louvain = nx.community.modularity(U, com, weight="peso")
    pais = dict(zip(clubes["clube_id"], clubes["pais"]))
    por_pais = {}
    for v in U:
        por_pais.setdefault(pais[v], set()).add(v)
    q_pais = nx.community.modularity(U, list(por_pais.values()), weight="peso")
    log(f"{len(com)} comunidades | modularidade Louvain = {q_louvain:.3f} | "
        f"partição por país = {q_pais:.3f}")
    linhas = []
    for i, c in enumerate(sorted(com, key=len, reverse=True)):
        cont = Counter(pt(pais[v]) for v in c)
        dom, n = cont.most_common(1)[0]
        linhas.append({"comunidade": i, "tamanho": len(c), "pais_dominante": dom, "pureza": n / len(c),
                       "composicao": "; ".join(f"{p} {k}" for p, k in cont.most_common(5))})
    df = pd.DataFrame(linhas)
    df.to_csv(TAB / "a6_comunidades.csv", index=False)
    pureza_media = (df["pureza"] * df["tamanho"]).sum() / df["tamanho"].sum()
    log(f"Pureza média (fração de clubes do país dominante): {pureza_media:.1%}")
    multi = df[df["pureza"] < 0.8]
    log(f"Comunidades multinacionais (pureza < 80%): {len(multi)}")
    for _, r in multi.head(6).iterrows():
        log(f"  [{r['tamanho']:3} clubes] {r['composicao']}")


# ---------------------------------------------------------------- A7
def a7_saldo_paises(transf):
    log("\n== A7. Saldo de jogadores e dinheiro por país (transferências internacionais) ==")
    t = transf[transf["pais_origem"] != transf["pais_destino"]]
    saiu = t.groupby("pais_origem").agg(saidas=("player_id", "size"), receita=("valor_eur", "sum"))
    entrou = t.groupby("pais_destino").agg(entradas=("player_id", "size"), despesa=("valor_eur", "sum"))
    df = saiu.join(entrou, how="outer").fillna(0)
    df["saldo_jogadores"] = df["entradas"] - df["saidas"]
    df["saldo_financeiro_mi_eur"] = (df["receita"] - df["despesa"]) / 1e6
    df = df.sort_values("saldo_financeiro_mi_eur")
    df.to_csv(TAB / "a7_saldo_por_pais.csv")
    for p, r in df.iloc[list(range(4)) + list(range(-6, 0))].iterrows():
        log(f"  {pt(p):14} saldo financeiro {r['saldo_financeiro_mi_eur']:9.0f} mi€   "
            f"saldo jogadores {r['saldo_jogadores']:6.0f}")

    d = df.copy()
    d["nome"] = [pt(p) for p in d.index]
    fig, ax = plt.subplots(figsize=(3.5, 4.4))
    cores = ["#2a78d6" if p in TOP5 else "#eb6834" if p in AMERICA_SUL else "#1baf7a" for p in d.index]
    ax.barh(d["nome"], d["saldo_financeiro_mi_eur"], color=cores, height=0.7)
    ax.axvline(0, color="#c3c2b7", linewidth=0.8)
    ax.set_xlabel("Receita − despesa com transferências internacionais (milhões de €)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=r) for r, c in COR.items()], frameon=False, loc="center left")
    salvar(fig, "a7_saldo_financeiro_paises")


# ---------------------------------------------------------------- A8
def a8_escalabilidade(arestas):
    log("\n== A8. Escalabilidade do Brandes ==")
    rng = random.Random(SEMENTE)
    nos = sorted(set(arestas["origem"]) | set(arestas["destino"]))
    linhas = []
    REPETICOES = 3   # 3 amostras por tamanho; usa-se a mediana do tempo (reduz ruído do SO)
    for k in [100, 200, 300, 400, 500, 600, 700, len(nos)]:
        for rep in range(REPETICOES):
            amostra = set(rng.sample(nos, k))
            sub = arestas[arestas["origem"].isin(amostra) & arestas["destino"].isin(amostra)]
            E = list(sub[["origem", "destino", "n_transferencias"]].itertuples(index=False, name=None))
            t0 = time.perf_counter()
            brandes(adjacencia_de_arestas(E))
            linhas.append({"V": k, "E": len(E), "tempo_s": time.perf_counter() - t0})
    df = pd.DataFrame(linhas).groupby("V").agg(E=("E", "mean"), tempo_s=("tempo_s", "median")).reset_index()
    df["V_E"] = df["V"] * df["E"]
    df.to_csv(TAB / "a8_escalabilidade.csv", index=False)
    coef = np.polyfit(df["V_E"], df["tempo_s"], 1)
    r2 = np.corrcoef(df["V_E"], df["tempo_s"])[0, 1] ** 2
    log(df.to_string(index=False))
    log(f"Ajuste linear tempo ~ V*E: R² = {r2:.3f}")

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(df["V_E"] / 1e6, df["tempo_s"], color="#2a78d6", linewidth=2, marker="o", markersize=5,
            label="Tempo medido")
    x = np.linspace(0, df["V_E"].max(), 50)
    ax.plot(x / 1e6, np.polyval(coef, x), color=MUTED, linewidth=1, linestyle="--",
            label=f"Ajuste linear em V·E (R² = {r2:.3f})")
    for _, r in df[df["V"] >= 400].iterrows():
        ax.annotate(f"V={int(r['V'])}", (r["V_E"] / 1e6, r["tempo_s"]), fontsize=5.5, color=INK2,
                    xytext=(-4, 6), textcoords="offset points", ha="right")
    ax.set_xlabel("V · E (milhões)")
    ax.set_ylabel("Tempo de execução (s)")
    ax.legend(frameon=False, loc="upper left")
    salvar(fig, "a8_escalabilidade")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    clubes, arestas, transf = carregar()
    G = grafo_nx(arestas)
    a1_intermediacao_global(arestas, clubes, G)
    a2_clubes_ponte(arestas, clubes)
    a3_caminhos_reais(clubes)
    a4_temporal(transf, clubes)
    a5_pagerank(G, clubes)
    a6_comunidades(G, clubes)
    a7_saldo_paises(transf)
    a8_escalabilidade(arestas)
    (RAIZ / "results" / "resumo.txt").write_text("\n".join(resumo), encoding="utf-8")


if __name__ == "__main__":
    main()
