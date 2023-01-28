# Python IA Visão Computacional

Trabalho para a matéria de visão computacional do curso Inteligência Artificial Aplicada da UFPR,
que consiste em realizar o treinamento e teste de imagens de dentes saudáveis (hg) e cáries com as
seguintes classificações: c2,c3,c4,c5,c6.

## Requisitos

Python 3.10.3

## Instalação

Criar o virtualenv conforme [padrão](https://docs.python.org/pt-br/3/library/venv.html)
ou se você utiliza o `asdf`, somente adicioná-lo localmente ou globalmente, mais detalhes neste
[link](https://github.com/asdf-community/asdf-python).

Executar a instalação do arquivo `requirements.txt`

```shell
$ pip install - r requirements.txt
```

## Utilização

Por default já esta selecionado para utilização do melhor modelo que foi escolhido,
no caso, RandomForest.

Altere o arquivo new_images.csv com o `path` delas e a classificação desejada. 
Execute o script conforme exemplo abaixo e compare a classificação gerada com as do arquivo:

```shell
$ python main.py
```

## Exemplo

Arquivo `new_images.csv`:

```csv
path,class
banco_dentes_IAA/c2/0000000015_c2.jpg,c2
banco_dentes_IAA/c2/0000000017_c2.jpg,c2
banco_dentes_IAA/hg/0000000006_hg.jpg,hg
```

Saída terminal:

```shell
$ python main.py
[...]
['c2' 'c2' 'hg']
```

## Treinamento

Descomente as linhas 79 à 93 do arquivo `main.py` e comente as linhas 95 à 100.

Mantenha a estrutura de pastas `banco_dentes_IAA/<CLASSIFICAÇÃO>` e adicione as imagens 
e execute o script conforme exemplo no tópico `Utilização`. 

Será realizado o treinamento de 80% do range dessas imagens e realizado teste de 20%.
Apresentando os seguintes indicares para escolha de melhor modelo: 
accuracy, precision, recall, f1.