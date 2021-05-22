
# Trabalho Final - Gerador de Legenda 📝
### Disciplina: TÓPICOS ESPECIAIS DE ENGENHARIA DE SOFTWARE 💻
### Professor: DIEGO LUIZ DORGAM AGUILERA 
### ImageCaptioning
## Participantes:
- Andre Lucas de Sousa Pinto -  170068251
-  Joao Gabriel Rossi de Borba -  170013693
- Julia de Melo Franco Fernandes - 170037738
- Lieverton Santos Silva - 170039251


## :one: Descrição do Projeto
- O projeto se trata de uma rede neural geradora de legenda;
- O código recebe uma foto e, a partir dela, gera uma legenda se baseando nos seus detalhes;

## :two: Pré-requisitos

Antes de começar, verifique se você atendeu aos seguintes requisitos:
<!---Estes são apenas requisitos de exemplo. Adicionar, duplicar ou remover conforme necessário--->
* Você instalou a versão mais recente do `Python`
* Você tem uma máquina `<Windows / Linux / Mac>`.
* Adicionou o dataset na mesma pasta do código.
* Você leu `<guia / link / documentação_relacionada_ao_projeto>`.

## :three: Especificações Técnicas
Para criar o projeto, foi necessária a utilização de 2 redes neurais, sendo elas:
- CNN (extrator de features);
- RNN (processador dos dados gerais);
 
### :four: CNN - Convolutional Neural Network
- Para a CNN foi utilizado um modelo pre-treinado e retirada a ultima camada do mesmo, pois o objetivo era extrair as features encontradas nas imagens e reduzir as dimensões, facilitando a geração de legenda da RNN;
- Foi utilizado o modelo InceptionV3;
 
### :five: RNN - Recurrent Neural Networks
- A RNN começa com uma camada de Embedding para trata das diferentes palavras do vocabulário;
- As camadas recorrentes da RNN são do tipo LSTM (long-short term memory) ao invés de GRU ou uma célula RNN padrão;

###  :six: Dataset
Dados retirados desse link:
-  [Dataset](https://github.com/jbrownlee/Datasets)
