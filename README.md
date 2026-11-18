# Previsão de preços de casas em Lisboa

Modelo de machine learning que estima o preço pedido de apartamentos à venda em Lisboa (até cerca de 2,2 M€), treinado com [AutoGluon](https://auto.gluon.ai/) a partir de anúncios do website **idealista**. O modelo prevê diretamente o preço em euros e é treinado para reduzir o erro médio em euros (MAE).

Todo o projeto está num só notebook, [`house_prices.ipynb`](house_prices.ipynb), que segue as fases de um projeto de ML:

1. Definição do problema
2. Recolha de dados
3. Escolha das variáveis
4. Limpeza dos dados
5. Análise exploratória (EDA)
6. Novas variáveis (feature engineering)
7. Divisão em treino e teste
8. Outliers (regra do IQR)
9. Valores em falta
10. Codificação e normalização
11. Modelo de referência (baseline)
12. Treino com AutoGluon
13. Avaliação no teste
14. Análise de erros
15. Importância das variáveis
16. Previsão para uma casa nova
17. Conclusões e limitações

## Resultados

No conjunto de teste (284 casas que o modelo nunca viu):

| Modelo | MAE | MAPE | Erro mediano | RMSE | R² |
|---|---|---|---|---|---|
| Baseline (€/m² da freguesia × área) | 190 160 € | 23,1% | 18,6% | 269 756 € | 0,59 |
| **AutoGluon** | **121 041 €** | **14,7%** | **10,6%** | **181 030 €** | **0,82** |

- O AutoGluon reduz o erro médio em euros em 36% face à baseline.
- Metade das casas tem um erro abaixo de 10,6% (erro mediano).
- São comparados 6 tipos de modelo: LightGBM, CatBoost, XGBoost, Random Forest, Extra Trees e uma rede neuronal. O notebook mostra as métricas de cada um no teste. O AutoGluon escolhe o modelo final pela validação, durante o treino, sem olhar para o teste. O modelo final, uma combinação dos melhores, é também o melhor no teste; sozinho, o melhor é o CatBoost.
- As variáveis mais importantes são a área, as casas de banho, a distância ao centro (Marquês de Pombal) e a garagem. A localização aparece repartida por várias variáveis (distância ao centro, coordenadas e freguesia) e, somadas, pesam bastante.

Cuidados que fazem diferença no resultado:

- **Fuga de informação:** ficam de fora as colunas calculadas a partir do preço, incluindo o selo de luxo do idealista, que só aparece em casas acima de 1 M€.
- **Casas repetidas:** a mesma casa anunciada por várias agências foi removida (128 linhas). Assim não fica uma cópia no treino e outra no teste.
- **Áreas impossíveis:** casas com menos de 20 m² são erros nos anúncios e foram removidas (4 casas).
- **Outliers:** casas acima de Q3 + 1,5 × IQR do preço (cerca de 2,2 M€) ficam de fora. O limite é calculado só com o treino e aplicado ao treino e ao teste.
- **Valores em falta:** nas colunas sim/não (elevador, garagem, piscina…), guardadas como 1/0, um anúncio sem valor conta como 0 (não tem). No andar e nas casas de banho usa-se a mediana, calculada só com o treino depois da divisão treino/teste. No estado, fica "desconhecido".

## Dados

Anúncios de apartamentos à venda em Lisboa, recolhidos do website idealista: 1 661 anúncios e 453 colunas. Ficam 1 533 casas depois de tirar as repetidas, 1 529 depois de tirar as áreas impossíveis e 1 422 depois de tirar os outliers.

O CSV **não está no repositório**, porque inclui contactos de anunciantes. Para correr o notebook, coloca o ficheiro na pasta do projeto com o nome `dataset_2026-10-12.csv`.

## Como correr

Requer Python 3.13.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
python -m jupyter lab house_prices.ipynb
```

O treino demora até cerca de 15 minutos num portátil sem GPU. Os modelos de árvores dão sempre os mesmos resultados; a rede neuronal pode variar ligeiramente de computador para computador e, com ela, as métricas do modelo final.

## Estrutura

```
├── house_prices.ipynb   # notebook com todas as fases e resultados
├── requirements.txt     # dependências com versões fixas
├── .gitignore           # ficheiros que não vão para o repositório (dados, modelos treinados)
└── README.md
```
