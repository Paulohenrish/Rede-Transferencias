"""
Etapa 1 - Pré-processamento.

Lê os CSVs brutos do dataset "transfermarkt-datasets" (Kaggle: davidcariboo/player-scores)
e gera o dataset tratado usado na modelagem do grafo:

  data/processed/clubes.csv          -> vértices (clube, país, liga, região)
  data/processed/transferencias.csv  -> transferências filtradas (2010-2026, entre clubes conhecidos)
  data/processed/arestas.csv         -> arestas agregadas (origem, destino, nº de transferências, valor total)

Decisões de limpeza (documentadas no artigo):
  1. Só entram clubes presentes em clubs.csv (796 clubes das 32 ligas acompanhadas pela base),
     pois são os únicos com país/liga conhecidos.
  2. Janela temporal: transferências com data entre 2010-01-01 e 2026-12-31.
  3. Removidas transferências de um clube para ele mesmo.
  4. Clubes da liga colombiana (COL1) não constam em competitions.csv; o país é atribuído manualmente.
  5. Valor (transfer_fee) ausente é mantido como ausente; o peso principal das arestas é a contagem.

Uso:  python3 src/preprocess.py
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
BRUTO = RAIZ / "data" / "raw"
TRATADO = RAIZ / "data" / "processed"

ANO_INICIO, ANO_FIM = 2010, 2026

TOP5 = {"England", "Spain", "Italy", "Germany", "France"}
AMERICA_SUL = {"Brazil", "Argentina", "Colombia"}  # únicas ligas sul-americanas cobertas pela base

# ligas que aparecem em clubs.csv mas não em competitions.csv
LIGAS_SEM_CADASTRO = {"COL1": "Colombia"}


def regiao(pais: str) -> str:
    if pais in TOP5:
        return "Top 5 Europa"
    if pais in AMERICA_SUL:
        return "América do Sul"
    return "Outros"


def main() -> None:
    TRATADO.mkdir(parents=True, exist_ok=True)

    clubes = pd.read_csv(BRUTO / "clubs.csv")
    comps = pd.read_csv(BRUTO / "competitions.csv")
    transf = pd.read_csv(BRUTO / "transfers.csv")

    # ---- vértices: clube -> país via liga doméstica ----
    clubes = clubes.merge(
        comps[["competition_id", "country_name"]],
        left_on="domestic_competition_id", right_on="competition_id", how="left",
    )
    clubes = clubes.rename(columns={"club_id": "clube_id", "name": "nome",
                                    "country_name": "pais", "domestic_competition_id": "liga"})
    clubes["pais"] = clubes["pais"].fillna(clubes["liga"].map(LIGAS_SEM_CADASTRO))
    assert clubes["pais"].notna().all(), "há clubes sem país definido"
    clubes["regiao"] = clubes["pais"].map(regiao)
    clubes = clubes[["clube_id", "nome", "pais", "liga", "regiao"]]
    ids_validos = set(clubes["clube_id"])

    # ---- transferências ----
    n_bruto = len(transf)
    transf["data"] = pd.to_datetime(transf["transfer_date"], errors="coerce")
    transf = transf[transf["data"].dt.year.between(ANO_INICIO, ANO_FIM)]
    n_janela = len(transf)
    transf = transf[transf["from_club_id"].isin(ids_validos) & transf["to_club_id"].isin(ids_validos)]
    n_conhecidos = len(transf)
    transf = transf[transf["from_club_id"] != transf["to_club_id"]]

    pais = dict(zip(clubes["clube_id"], clubes["pais"]))
    transf = transf.assign(
        origem=transf["from_club_id"], destino=transf["to_club_id"],
        pais_origem=transf["from_club_id"].map(pais), pais_destino=transf["to_club_id"].map(pais),
        valor_eur=transf["transfer_fee"],
    )[["player_id", "player_name", "data", "transfer_season", "origem", "destino",
       "pais_origem", "pais_destino", "valor_eur"]].sort_values(["player_id", "data"])

    # ---- arestas agregadas ----
    arestas = (transf.groupby(["origem", "destino"])
               .agg(n_transferencias=("player_id", "size"),
                    valor_total_eur=("valor_eur", lambda s: s.fillna(0).sum()))
               .reset_index())

    clubes.to_csv(TRATADO / "clubes.csv", index=False)
    transf.to_csv(TRATADO / "transferencias.csv", index=False)
    arestas.to_csv(TRATADO / "arestas.csv", index=False)

    print(f"Transferências brutas:                 {n_bruto:>7}")
    print(f"  na janela {ANO_INICIO}-{ANO_FIM}:              {n_janela:>7}")
    print(f"  entre clubes com país conhecido:     {n_conhecidos:>7}")
    print(f"  após remover laços (final):          {len(transf):>7}")
    print(f"  sem valor informado:                 {transf['valor_eur'].isna().mean():>7.1%}")
    print(f"Vértices (clubes): {len(clubes)}   Arestas (pares origem->destino): {len(arestas)}")


if __name__ == "__main__":
    main()
