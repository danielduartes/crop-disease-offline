# Crop Disease Offline

Classificador de doenças em folhas (15 classes de tomate, batata e pimentão) feito com **transfer learning** (MobileNetV2) e exportado para rodar **no navegador, sem internet**, com TensorFlow.js.

O objetivo do projeto era provar o **fluxo de ponta a ponta**: treinar em Python, converter e rodar offline numa página. Ele **não entrega um detector utilizável na plantação**: o modelo vai bem no dataset e errou quase todas as fotos de folhas em ambiente real que testei (ver "Fotos de fora do dataset").

## Resumo dos resultados

- **F1 macro de 0,873 e acurácia de 0,90 no teste**, em fotos do mesmo dataset (PlantVillage).
- **Fotos de fora do dataset:** das 8 folhas de cultura conhecida que supus saudáveis, o modelo acertou 1. As outras foram classificadas como doentes, algumas com mais de 90% de confiança.
- **O fundo carrega informação sobre a classe:** um classificador que só vê a cor das bordas, sem a folha, acerta 38% (chute: 7%).
- **A conversão para o navegador deu trabalho:** exigiu um ambiente separado (Python 3.11, caminho curto) e levar só os pesos entre versões do Keras. A página responde como o Python nas duas fotos que comparei.

## Dataset

PlantVillage, versão de 15 classes do Kaggle ([emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease/data), a confirmar se foi esta a que usei). Verificar a licença na página do dataset antes de redistribuir qualquer imagem.

20.639 arquivos (20.636 `.jpg`, 1 `.png`, 1 `.jpeg` e 1 sem extensão), em uma pasta por classe:

| Classe | Imagens |
|---|---|
| `Pepper__bell___Bacterial_spot` | 997 |
| `Pepper__bell___healthy` | 1.478 |
| `Potato___Early_blight` | 1.000 |
| `Potato___Late_blight` | 1.000 |
| `Potato___healthy` | 152 |
| `Tomato_Bacterial_spot` | 2.127 |
| `Tomato_Early_blight` | 1.000 |
| `Tomato_Late_blight` | 1.909 |
| `Tomato_Leaf_Mold` | 952 |
| `Tomato_Septoria_leaf_spot` | 1.771 |
| `Tomato_Spider_mites_Two_spotted_spider_mite` | 1.676 |
| `Tomato__Target_Spot` | 1.404 |
| `Tomato__Tomato_YellowLeaf__Curl_Virus` | 3.209 |
| `Tomato__Tomato_mosaic_virus` | 373 |
| `Tomato_healthy` | 1.591 |

Observações sobre os dados:

- **Desbalanceamento forte:** a batata saudável tem 152 imagens e o vírus do tomate 3.209 (21 vezes mais). Tomate é 78% do dataset.
- **Folhas isoladas em fundo liso**, em condições controladas. É o que explica boa parte do problema em campo.
- **Os nomes de arquivo têm o padrão `UUID___CÓDIGO NÚMERO.JPG`.** O código parece identificar a fonte ou a sessão da foto (a interpretação é minha, não está documentada). Quase todo código aparece em uma classe só, com uma exceção (`RS_HL`, em batata saudável e tomate saudável).
- **Quatro arquivos fora do padrão** (um sem extensão, um `.png`, dois com nomes de outra origem) ficaram de fora da divisão, e não investiguei o que são.

### O fundo como possível atalho

Medi a cor média das 4 bordas de cada foto. Ela varia bastante entre fontes (de uns 110 a uns 163 em 255), e a cor média das bordas varia até dentro da mesma classe. Treinei um `RandomForest` que vê **só 12 números** (a cor de cada borda), sem a folha:

| | Acurácia |
|---|---|
| Só a cor das bordas | **0,38** (F1 macro 0,30) |
| Sempre a classe mais comum | 0,13 |
| Chute aleatório entre 15 classes | 0,07 |

Isso mostra que o fundo e a classe estão ligados nos dados. Não prova que a MobileNetV2 usa esse atalho, e um modelo com acesso à imagem inteira pode usá-lo mais ou menos do que essa medida mostra.

## Divisão treino, validação e teste

O dataset não vem dividido. Fotos de números vizinhos se parecem muito (vi isso na `Tomato_Early_blight`), então um sorteio por foto colocaria quase-gêmeas dos dois lados e inflaria a validação.

Em vez disso, dentro de **cada código de fonte**, ordenei as fotos pelo número do nome, cortei em **blocos de 10 consecutivas** e sorteei os blocos: 70% treino, 15% validação e 15% teste.

| | Imagens |
|---|---|
| Treino | 14.324 |
| Validação | 3.063 |
| Teste | 3.055 |

Ficaram de fora uns 196 arquivos (1%): pelas contagens, códigos com menos de 5 fotos, mas não conferi quais foram. O teste da `Potato___healthy` tem só **20 imagens**, então o recall dela oscila muito.

O bloco de 10 é uma precaução: **não verifiquei** se fotos vizinhas são mesmo a mesma folha, nem se a mesma folha aparece em códigos diferentes.

## Modelo

- **MobileNetV2** com pesos do ImageNet, imagens 160×160, lotes de 32, `preprocess_input` do próprio modelo (pixels entre -1 e 1).
- **Camada nova:** `GlobalAveragePooling2D`, `Dropout(0,3)` e `Dense(15, softmax)`. O modelo todo tem 2.277.199 parâmetros.
- **Augmentation só no treino:** rotação de até 30°, espelhamento horizontal e vertical, zoom de 20% e brilho entre 0,7 e 1,3 (uma tentativa de reduzir a dependência do fundo; não testei o efeito isolado).
- **Pesos de classe suaves:** a raiz do peso balanceado (`total / (15 × imagens da classe)`). A batata saudável vai de 8,53 para 2,92, e o vírus do tomate de 0,42 para 0,65. Foi uma escolha minha, e **não testei o peso cheio** para comparar.
- **Fase 1:** base congelada, 5 epochs (22,6 min em CPU). Só a camada final aprende.
- **Fase 2 (fine-tuning):** últimas 30 camadas da base liberadas (as de normalização continuam congeladas), taxa de aprendizado `1e-5`, 5 epochs (25,8 min).
- **Escolha do modelo:** um callback calcula o **F1 macro** na validação a cada epoch e salva o melhor. Foi a métrica principal, escolhida antes de ver o teste, porque a acurácia é puxada pelas classes grandes.

## Resultados

### Validação (3.063 imagens)

| | F1 macro | Acurácia |
|---|---|---|
| Fase 1 (base congelada) | 0,835 | 0,858 |
| Fase 2 (fine-tuning, epoch 4) | **0,888** | 0,902 |

O fine-tuning melhorou o F1 em 14 das 15 classes e manteve uma igual (o vírus do tomate, 0,97). Destaques:

| Classe | F1 (fase 1 → 2) | Recall | Precision |
|---|---|---|---|
| `Tomato_Early_blight` | 0,49 → 0,70 | 0,34 → 0,59 | 0,86 → 0,84 |
| `Tomato__Tomato_mosaic_virus` | 0,83 → 0,86 | 0,72 → 0,98 | 0,98 → 0,76 |

No mosaico, a precision caiu porque o modelo passou a prever a classe com mais frequência (uns 18 falsos positivos, vindos de várias classes, sem uma dominante).

**Versão normal ou ajustada pelos pesos.** Testei dividir as probabilidades pelos pesos (como no meu projeto do lixo): F1 macro de 0,893 contra 0,888. A regra que defini antes era só adotar a ajustada se passasse por 0,01, então ficou a normal.

### Teste (3.055 imagens, usado uma vez)

F1 macro **0,873** e acurácia **0,90**, um pouco abaixo da validação, como esperado (o checkpoint foi escolhido por ela). Os pontos fracos:

| Classe | F1 | Observação |
|---|---|---|
| `Tomato_Early_blight` | 0,67 | recall 0,55, a pior classe, como na validação |
| `Potato___healthy` | 0,73 | recall 1,00 e precision 0,57, com 20 imagens |
| `Tomato__Target_Spot` | 0,78 | |
| as outras 12 | 0,84 a 0,98 | |

Validação e teste saem do mesmo processo (mesmos códigos de fonte, mesmo tipo de fundo, só em blocos diferentes). Por isso o resultado mede a generalização **entre blocos do dataset**, e não para fotos tiradas em campo.

### Onde a `Tomato_Early_blight` erra

Na validação, dos erros dessa classe, 25 vão para `Septoria_leaf_spot` e 14 para `Late_blight`. Em 8 imagens de cada grupo, as acertadas tinham dano extenso (áreas grandes escuras), e as chamadas de Septoria tinham poucas manchas isoladas. Minha hipótese é que o modelo reconhece o estágio avançado e erra o inicial. São 8 imagens sorteadas, um indício e não uma prova. Não testei imagens maiores que 160×160.

## Fotos de fora do dataset

Testei 12 fotos da internet (várias com marca d'água, algumas recortadas em fundo liso). **Não estão no repositório**, porque têm dono. Supus o rótulo pela cultura que as fotos mostravam, **sem confirmação de um especialista**.

Das 8 folhas de cultura conhecida que supus saudáveis (7 de tomate e pimentão, 1 de batata):

- **1 acerto:** um recorte de pimentão em fundo branco (99,9%), que se parece com o dataset.
- **7 erros**, quase sempre apontando para uma doença de manchas (`Early_blight`, `Septoria`, `Late_blight`). Um ramo de tomate foi chamado de `Early_blight` com **95%**, um erro seguro de si.
- As folhas em fundo liso foram as únicas com confiança alta e correta. As com terra, mão, janela ou outras folhas erraram.
- Três ou quatro fotos pareciam **batata-doce**, que o modelo nunca viu. Uma delas foi chamada de pimentão saudável com 100%: **o modelo não sabe dizer "não conheço isso"**.

São 12 fotos escolhidas por mim, e isso ilustra o problema sem medi-lo: não calculei taxa de acerto. O resultado é coerente com o fundo servir de atalho, e com o modelo ter aprendido que "folha em ambiente natural" parece "doente", mas não testei essas hipóteses.

## Exportação para o navegador

O modelo salvo em Keras 3 (`.keras`) **não abre no Keras 2.15**, que é o que o conversor do TensorFlow.js aceitou neste ambiente. O fluxo que funcionou:

1. `exportar_pesos.py` (ambiente principal): carrega o modelo e salva os 262 arrays de pesos em um `.npz`, que não depende de versão do Keras.
2. `converter.py` (ambiente de conversão): reconstrói a mesma arquitetura em Keras 2.15, carrega os pesos, **confere o número e a forma de cada array**, e converte para `web/modelo/`.
3. **Conferência de saída:** a mesma entrada fixa dá, nos dois ambientes, os mesmos 5 primeiros valores até uns 5 dígitos significativos (por exemplo, `8.2113424e-08` contra `8.2114674e-08`). Só as formas iguais não provariam que os pesos foram para as camadas certas.

O ambiente de conversão exigiu:

- **Python 3.11.** O TensorFlow 2.15 não tem instalador para o 3.14.
- **Um caminho curto, fora do OneDrive** (`C:\venv_conv`). No caminho longo, a instalação do TensorFlow ficou incompleta (o `import tensorflow` carregava uma pasta vazia, sem `__version__`). A causa é uma suspeita minha, e não consegui isolá-la.
- **`tensorflowjs==4.17.*` com `--no-deps`**, porque a dependência `uvloop` não existe para Windows.
- **Módulos falsos** para `tensorflow_decision_forests` e `jax` no começo do `converter.py`: o conversor os importa no topo, mas não usa para um modelo Keras. Não usei o conversor em nenhum outro tipo de modelo.

Saída: `model.json` (136 KB) e 3 arquivos de pesos, uns **9,2 MB** no total.

## Página (`web/`)

Um `index.html` carrega o modelo, deixa escolher uma foto, redimensiona para 160×160, aplica o mesmo pré-processamento do treino (`x / 127,5 − 1`) e mostra as 3 classes mais prováveis. A biblioteca (`tf.min.js`) fica no disco, sem CDN. Há um **aviso visível** de que é uma demonstração técnica e não um diagnóstico.

Comparei duas fotos com o Python:

| Foto | Python | Página |
|---|---|---|
| recorte de pimentão | `Pepper__bell___healthy` 1,00 | `Pepper__bell___healthy` 99,9% |
| ramo de tomate | `Tomato_Early_blight` 0,96 | `Tomato_Early_blight` 95,4% |

**Teste offline:** [preencher: rede aberta com todas as requisições para `localhost`? Wi-Fi desligado e a classificação continuou funcionando?]. Isso prova que roda sem internet **nesta máquina e com o servidor local** (`python -m http.server`). Uma página que abra sem servidor, ou instalável no celular, não foi testada.

## O que aprendi

- **Um número alto no dataset não diz nada sobre o campo.** 90% no teste e 1 acerto em 8 fotos reais, com o mesmo modelo. A medida que importa para o uso é a que se parece com o uso.
- **Em dataset assim, a divisão importa tanto quanto o modelo.** Fotos vizinhas e fundos por fonte podem inflar a validação sem nenhum erro aparecer.
- **A média esconde a classe fraca.** O F1 macro subiu e a `Tomato_Early_blight` continuou a pior.
- **Uma média de classes pequenas oscila:** com 20 imagens, cada erro mexe 5 pontos no recall.
- **Escolher a métrica e a versão do modelo antes de ver o teste** é o que permite chamar aquele número de honesto.
- **O ecossistema do TensorFlow.js anda atrás do TensorFlow.** Exportar o modelo foi o passo mais frágil do projeto, e envolveu seis problemas de versão em sequência.

## Limitações

- **O modelo não serve como detector em campo.** Não o usaria para decidir tratamento de plantas.
- **Só conhece 15 classes** e não tem a opção "não é uma dessas": culturas desconhecidas viram uma das 15.
- **O teste de campo é pequeno** (12 fotos), de internet, com rótulos supostos por mim.
- **A hipótese do atalho do fundo** é apoiada pelo teste das bordas e pelas fotos de campo, mas não foi testada no modelo final (por exemplo, trocando o fundo das imagens).
- **Treino em CPU** (sem GPU no Windows), 5 epochs por fase. A validação ainda oscilava entre epochs.
- **Os 196 arquivos descartados** e os 4 fora do padrão não foram investigados.
- **O número de camadas liberadas (30)** veio de outro projeto meu e não foi ajustado aqui.

## Próximos passos possíveis

- Testar **224×224** para ver se lesões pequenas da `Early_blight` melhoram (custaria uns 2 vezes o tempo de treino, estimativa minha).
- **Trocar o fundo** das imagens de treino (recortar a folha e colar em fundos variados) e medir nas fotos de campo.
- Treinar com um dataset que **tenha fotos de campo**.
- Uma classe "não é folha de uma dessas culturas", ou um limiar de confiança.
- Rodar a página sem servidor local ou instalável no celular.

## Estrutura

```
crop-disease-offline/
├── [notebook].ipynb      # dados, divisão, treino, avaliação e testes de campo
├── exportar_pesos.py     # Keras 3: modelo -> pesos (.npz)
├── converter.py          # Keras 2.15: pesos -> TensorFlow.js
├── web/
│   ├── index.html
│   ├── tf.min.js
│   └── modelo/           # model.json + 3 arquivos .bin
├── data/                 # dataset (não versionado)
├── modelos/              # .keras e .npz (não versionado)
├── fotos_campo/          # fotos de teste (não versionado)
└── README.md
```

## Como executar

**Só a página (sem treinar):**

```
cd web
python -m http.server 8000
```

Abra `http://localhost:8000`. O servidor local é necessário porque o navegador bloqueia o carregamento do modelo a partir de `file://`.

**Treino:** baixe o dataset em `data/PlantVillage`, instale `tensorflow matplotlib scikit-learn` e rode o notebook por seção. As células que copiam arquivos (a divisão em `data/dividido`) só precisam rodar uma vez, e o treino salva os modelos em `modelos/`.

**Conversão:** gere os pesos com `python exportar_pesos.py` no ambiente do treino. Depois, num ambiente com **Python 3.11** em caminho curto, instale `tensorflow==2.15.*`, `numpy<2` e `tensorflowjs==4.17.*` (com `--no-deps`) e rode `python converter.py`. Não testei se `tensorflow-hub`, `tensorflow-decision-forests` e `pandas` (que cheguei a instalar) são necessários depois dos módulos falsos.

No Windows, o TensorFlow atual não usa GPU: cada epoch leva uns 4 a 5 minutos na CPU.