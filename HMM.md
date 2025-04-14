# Modelo Oculto de Markov

## Estados Ocultos e Observações

Considere uma informação que você não tem (estado oculto), mas consegue adivinhar ou inferir por meio de pistas
(observações).

**Por exemplo**:

Imagine que você quer saber como está o tempo lá fora, mas você não pode olhar pela janela.
Só que você tem um amigo que, todos os dias, vem te encontrar para ir trabalhar. Esse amigo pode
vir carregando um guarda-chuva ou não.

O estado oculto nessa situação é o tempo (se está chuvoso ou ensolarado), e a pista é a presença do
guarda-chuva.

## Conjunto de Estados

São todos os possíveis estados ocultos que o sistema pode assumir.

**Por exemplo**:

Na situação do tempo, o conjunto de estados seria: chuvoso ou ensolarado.

## Conjunto de Observações

São todas as possíveis pistas geradas a partir dos estados ocultos.

**Por exemplo**:

Na mesma situação, o estado oculto chuvoso gerará a observação "presença de guarda-chuva".

## Matriz de Transição

Uma matriz que tem as probabilidades de transição de um estado oculto para outro.

**Por exemplo**:

Se hoje está ensolarado, a probabilidade de amanhã continuar ensolarado é de 70%, enquanto para que
fique chuvoso é de 30%.

## Matriz de Emissão

Uma matriz que descreve a probabilidade de cada observação ocorrer, dado que o sistema está em um
determinado estado.

**Por exemplo**:

Se a pessoa está feliz, a matriz de emissão pode indicar que há 80% de chance de ela usar
palavras positivas ou emojis sorridentes; se estiver triste, as probabilidades podem indicar
maior chance de palavras negativas ou emojis tristes.

## Distribuição Inicial

É a definição das probabilidades com que o sistema começa em cada um dos estados possíveis.

**Por exemplo**:

Se, historicamente, a pessoa costuma começar o dia com bom humor, a distribuição inicial pode ter uma maior
probabilidade associada ao estado “feliz” no início do processo.

