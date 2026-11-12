# Clubes-ponte no mercado global de transferências

Análise em grafos do fluxo de jogadores sul-americanos (Brasil, Argentina e Colômbia) para as cinco grandes
ligas europeias. Trabalho final da disciplina de Estruturas de Dados (Árvores e Grafos) — CEFET-MG.

**Modelagem:** grafo dirigido e ponderado. Vértices são os 796 clubes das 32 ligas cobertas pela base;
existe aresta `u → v` se houve transferência de `u` para `v` entre 2010 e 2026. O peso é o número de
transferências, e a distância usada nos caminhos mínimos é `1 / peso`.

**Algoritmo principal:** centralidade de intermediação de Brandes (2001), implementada do zero em
`src/brandes.py` (Dijkstra com contagem de caminhos mínimos + acumulação de dependências), incluindo a
versão dirigida entre dois grupos (origens = clubes sul-americanos, alvos = clubes do Top 5).
A implementação é validada automaticamente contra o NetworkX.

## Requisitos

- Ubuntu 24.04 (ou qualquer Linux) com Python 3.10+
- Pacotes: `pandas`, `networkx`, `matplotlib`, `scipy`, `numpy`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dados

Baixe o dataset tratado do Zenodo (link no artigo) **ou** os arquivos brutos do
[transfermarkt-datasets](https://www.kaggle.com/datasets/davidcariboo/player-scores) e coloque
`transfers.csv`, `clubs.csv`, `competitions.csv` e `players.csv` em `data/raw/`.

## Como executar

```bash
./run.sh
```

ou, passo a passo:

```bash
python3 src/preprocess.py   # gera data/processed/ (clubes, transferências e arestas)
python3 src/analise.py      # roda as análises (~1 min) e gera results/
```

## Saídas

| Pasta | Conteúdo |
|---|---|
| `data/processed/` | `clubes.csv` (vértices), `arestas.csv` (arestas agregadas), `transferencias.csv` |
| `results/resumo.txt` | Todos os números citados no artigo |
| `results/tabelas/` | Uma tabela CSV por análise (A1 a A8) |
| `results/figuras/` | Gráficos em PDF (para o LaTeX) e PNG |

## Estrutura

```
src/preprocess.py   limpeza e montagem do dataset
src/brandes.py      algoritmo de Brandes (implementação própria)
src/analise.py      análises A1–A8 e gráficos
```
