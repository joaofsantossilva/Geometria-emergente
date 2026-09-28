# Geometria emergente em modelos físicos baseados em adjacência

Código da dissertação de mestrado em Física e Astrofísica, Faculdade de
Ciências da Universidade de Lisboa, 2026.

A dissertação estuda se a informação relacional de uma rede dá origem a
uma estrutura com carácter geométrico coexistente, aplicando uma
decomposição que reorganiza a rede num conjunto de massas em forma de
árvore e medindo sobre elas observáveis de fronteira. Os resultados são
comparados com redes sintéticas de referência construídas com a mesma
densidade de ciclos.

## Dados

Os dados de transações utilizados na dissertação **não são
distribuídos** neste repositório. Os programas esperam um ficheiro
`bq_BTC_results.csv` com as colunas `src` e `dst`, correspondentes aos
endereços de origem e destino de cada transação.

## Organização

### Análise

| ficheiro | o que faz |
|---|---|
| `sint_pa.py` | gera redes sintéticas com ligação preferencial ou uniforme e compara a decomposição nas duas, para testar se o algoritmo impõe por si próprio uma estatística de ligação |
| `btc_completo.py` | aplica a decomposição em massas à rede real pela regra de admissão por idade e calcula os observáveis de fronteira: paridade, casca, nulidade, classificação fermiónica, ciclos exportados e *backbone* |
| `pipeline_espectral.py` | calcula o espectro do Laplaciano da rede e das massas, a multiplicidade do valor próprio nulo, a separação entre os primeiros modos e a razão de participação inversa |
| `holonomia_btc.py` | mede a holonomia dos ciclos curtos da componente gigante: ângulo sólido do polígono esférico e fator spinorial, por comprimento de ciclo |
| `holonomia_nulo.py` | repete a medição anterior à escala de uma sub-rede, com um modelo de referência construído a partir da mesma sequência de graus |

### Figuras

| ficheiro | figura |
|---|---|
| `fig_sinal.py` | efeito da troca de sinal de um vetor próprio sobre a representação tridimensional |
| `fig_janela.py` | evolução dos observáveis com a duração da janela de observação |

## Dependências

Python 3, com `numpy`, `scipy`, `pandas`, `networkx` e `matplotlib`.

## Ordem de execução

Os programas de `codigo/` são independentes entre si e podem ser
corridos em qualquer ordem, desde que o ficheiro de dados esteja
disponível. O `sint_pa.py` é o único que não necessita dos dados reais,
por gerar as suas próprias redes.
