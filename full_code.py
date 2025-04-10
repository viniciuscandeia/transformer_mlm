import os

os.environ[ "KERAS_BACKEND" ] = "torch"

import keras_hub

import keras
from keras import layers
from keras.layers import TextVectorization

from dataclasses import dataclass
import pandas as pd
import numpy as np
import glob
import re
from pprint import pprint


@dataclass
class Config:
    MAX_LEN = 256
    BATCH_SIZE = 16  # 32
    LR = 0.0001  # 0.001
    VOCAB_SIZE = 30000
    EMBED_DIM = 128
    NUM_HEAD = 8  # used in bert model
    FF_DIM = 128  # used in bert model
    NUM_LAYERS = 1


config = Config()


def get_text_list_from_files( files ) -> list[ str ]:
    text_list: list[ str ] = [ ]
    for file_path in files:
        with open( file_path, "r", encoding = "utf-8" ) as file:
            text_list.append( file.read() )
    return text_list


def get_data_from_text_files( folder_name: str ) -> pd.DataFrame:
    files = glob.glob( f"musicas/{folder_name}/*.txt" )
    texts: list[ str ] = get_text_list_from_files( files )
    df = pd.DataFrame( { "lyric": texts } )
    df = df.sample( frac = 1 )
    df = df.reset_index( drop = True )
    return df


train_df = get_data_from_text_files( "train" )
test_df = get_data_from_text_files( "test" )

print( f"Tamanho do DataFrame de Treino: {len( train_df )}" )

print( f"Tamanho do DataFrame de Teste/Validação: {len( test_df )}" )

import tensorflow as tf


def custom_standardization( input_data ):
    lowercase = tf.strings.lower( input_data )
    stripped_html = tf.strings.regex_replace( lowercase, "<br />", " " )

    return tf.strings.regex_replace(
            stripped_html,
            "[%s]" % re.escape( "!#$%&'()*+,-./:;<=>?@\^_`{|}~" ),
            ""
    )


def get_vectorize_layer(
        texts: list[ str ], vocab_size: int, max_seq: int, special_tokens: list = [ "[MASK]" ]
) -> TextVectorization:
    vectorize_layer = TextVectorization(
            max_tokens = vocab_size,  # Define o número máximo de tokens no vocabulário.
            output_mode = "int",
            standardize = custom_standardization,
            output_sequence_length = max_seq,  # Define o comprimento fixo para todas as sequências de saída.
    )

    vectorize_layer.adapt( texts )
    vocab = vectorize_layer.get_vocabulary()
    vocab = vocab[ 2: vocab_size - len( special_tokens ) ] + special_tokens
    vectorize_layer.set_vocabulary( vocab )
    return vectorize_layer


train_texts = train_df.lyric.values.tolist()
vectorize_layer = get_vectorize_layer(
        train_texts,
        config.VOCAB_SIZE,
        config.MAX_LEN,
        special_tokens = [ "[mask]" ],
)

mask_token_id = vectorize_layer( [ "[mask]" ] ).cpu().numpy()[ 0 ][ 0 ]


def encode( texts: list[ str ] ) -> tf.Tensor:
    encoded_texts = vectorize_layer( texts )
    return encoded_texts.cpu().numpy()


def get_masked_input_and_labels( encoded_texts ):
    inp_mask = np.random.rand( *encoded_texts.shape ) < 0.15
    inp_mask[ encoded_texts <= 2 ] = False
    labels = -1 * np.ones( encoded_texts.shape, dtype = int )
    labels[ inp_mask ] = encoded_texts[ inp_mask ]
    encoded_texts_masked = np.copy( encoded_texts )
    inp_mask_2mask = inp_mask & (np.random.rand( *encoded_texts.shape ) < 0.90)
    encoded_texts_masked[ inp_mask_2mask ] = mask_token_id
    inp_mask_2random = inp_mask_2mask & (np.random.rand( *encoded_texts.shape ) < 1 / 9)
    encoded_texts_masked[ inp_mask_2random ] = np.random.randint(
            3, mask_token_id, inp_mask_2random.sum()
    )
    sample_weights = np.ones( labels.shape )
    sample_weights[ labels == -1 ] = 0
    y_labels = np.copy( encoded_texts )
    return encoded_texts_masked, y_labels, sample_weights


x_train = encode( train_df.lyric.values )
x_test = encode( test_df.lyric.values )

x_train_masked, y_train_labels, train_sample_weights = get_masked_input_and_labels(
        x_train
)
x_val_masked, y_val_labels, val_sample_weights = get_masked_input_and_labels(
        x_test
)

mlm_train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train_masked, y_train_labels, train_sample_weights)
)
mlm_train_ds = mlm_train_ds.shuffle( 1000 ).batch( config.BATCH_SIZE ).prefetch( tf.data.AUTOTUNE )
mlm_val_ds = tf.data.Dataset.from_tensor_slices(
        (x_val_masked, y_val_labels, val_sample_weights)
)
mlm_val_ds = mlm_val_ds.batch( config.BATCH_SIZE ).prefetch( tf.data.AUTOTUNE )


def bert_module( query, key, value, i ):
    attention_output = layers.MultiHeadAttention(
            num_heads = config.NUM_HEAD,  # Define o número de cabeças de atenção.
            key_dim = config.EMBED_DIM // config.NUM_HEAD,  # Define a dimensão de cada cabeça de atenção.
            name = f"encoder_{i}_multiheadattention",  # Nomeia a camada para fácil identificação.
    )( query, key, value )

    attention_output = layers.Dropout( 0.1, name = f"encoder_{i}_att_dropout" )(
            attention_output
    )
    attention_output = layers.LayerNormalization(
            epsilon = 1e-6, name = f"encoder_{i}_att_layernormalization"
    )( query + attention_output )
    ffn = keras.Sequential(
            [
                layers.Dense( config.FF_DIM, activation = "relu" ),
                layers.Dense( config.EMBED_DIM ),
            ],
            name = f"encoder_{i}_ffn",  # Nomeia a rede feed-forward.
    )

    ffn_output = ffn( attention_output )
    ffn_output = layers.Dropout( 0.1, name = f"encoder_{i}_ffn_dropout" )(
            ffn_output
    )

    sequence_output = layers.LayerNormalization(
            epsilon = 1e-6, name = f"encoder_{i}_ffn_layernormalization"
    )( attention_output + ffn_output )
    return sequence_output


loss_fn = keras.losses.SparseCategoricalCrossentropy( reduction = None )
loss_tracker = keras.metrics.Mean( name = "loss" )


class MaskedLanguageModel( keras.Model ):

    # Método para calcular a perda durante o treinamento ou avaliação
    # 'x' representa as entradas, 'y' os rótulos verdadeiros, 'y_pred' as previsões do modelo
    # e 'sample_weight' os pesos das amostras (usados aqui para focar nos tokens mascarados).
    def compute_loss( self, x = None, y = None, y_pred = None, sample_weight = None ):
        # Como a função de perda 'loss_fn' foi inicializada com 'reduction = None', ela retorna um tensor
        # contendo um valor de perda para cada exemplo no lote.
        # Queremos que a perda total do lote seja o valor a ser otimizado, então somamos todas as perdas individuais.
        loss = loss_fn( y, y_pred, sample_weight )

        # Atualiza o estado do rastreador de perda ('loss_tracker') com os valores de perda do lote atual.
        # O parâmetro 'sample_weight' garante que a média da perda seja ponderada corretamente pelos tokens que foram mascarados.
        loss_tracker.update_state( loss, sample_weight = sample_weight )

        # Retorna a soma total das perdas do lote. Este valor será usado pelo otimizador para ajustar os pesos do modelo.
        return keras.ops.sum( loss )

    # Método para calcular as métricas durante o treinamento ou avaliação
    # Recebe as entradas, os rótulos verdadeiros, as previsões e os pesos das amostras.
    def compute_metrics( self, x, y, y_pred, sample_weight ):
        # Retorna um dicionário contendo as métricas a serem monitoradas.
        # Neste caso, estamos retornando apenas a perda média calculada pelo 'loss_tracker'.
        return { "loss": loss_tracker.result() }

    # Propriedade para listar os objetos de métrica do modelo
    # Este método é importante para que o Keras possa gerenciar o estado das métricas automaticamente.
    # Ao listar os objetos 'Metric' aqui, o método 'reset_states()' será chamado automaticamente no início de cada época
    # de treinamento ou no início da função 'evaluate()'.
    # Se esta propriedade não for implementada, seria necessário chamar 'reset_states()' manualmente.
    @property
    def metrics( self ):
        return [ loss_tracker ]


def create_masked_language_bert_model():
    """
    Cria um modelo BERT para a tarefa de Masked Language Modeling (MLM).

    Este modelo consiste em uma camada de embedding para as palavras, uma camada de embedding para as posições,
    múltiplos blocos Transformer (definidos na função `bert_module`) e uma camada de classificação para prever os tokens mascarados.

    Returns:
        MaskedLanguageModel: Uma instância do modelo BERT configurada para MLM.
    """
    # Define a camada de entrada do modelo.
    # Espera-se que a entrada seja uma sequência de tokens (IDs inteiros) com comprimento máximo definido em 'config.MAX_LEN'.
    inputs = layers.Input( (config.MAX_LEN,), dtype = "int64" )

    # Cria a camada de embedding de palavras (word embeddings).
    # Esta camada transforma cada ID de token em um vetor denso de tamanho 'config.EMBED_DIM',
    # representando a semântica da palavra.
    word_embeddings = layers.Embedding(
            config.VOCAB_SIZE, config.EMBED_DIM, name = "word_embedding"
    )( inputs )

    # Cria a camada de embedding de posição (position embeddings) usando a biblioteca TensorFlow Hub.
    # Em modelos Transformer, a ordem das palavras é importante. Essa camada adiciona informações sobre a posição de cada token na sequência.
    # 'sequence_length' define o comprimento máximo da sequência para o qual os embeddings de posição serão gerados.
    position_embeddings = keras_hub.layers.PositionEmbedding(
            sequence_length = config.MAX_LEN
    )( word_embeddings )

    # Combina as embeddings de palavras e as embeddings de posição.
    # A soma dessas duas embeddings fornece ao modelo informações sobre o significado da palavra e sua posição na frase.
    embeddings = word_embeddings + position_embeddings

    # Inicializa a saída do encoder com a combinação das embeddings.
    encoder_output = embeddings

    # Aplica múltiplos blocos Transformer (definidos na função 'bert_module').
    # O número de camadas (blocos) é definido em 'config.NUM_LAYERS'.
    # Cada bloco Transformer realiza autoatenção multi-cabeças e uma rede feed-forward.
    for i in range( config.NUM_LAYERS ):
        encoder_output = bert_module( encoder_output, encoder_output, encoder_output, i )

    # Cria a camada de classificação para a tarefa de Masked Language Modeling (MLM).
    # Esta camada densa projeta a saída do encoder de volta para o tamanho do vocabulário ('config.VOCAB_SIZE').
    # A função de ativação 'softmax' transforma as saídas brutas da rede em um vetor de probabilidades,
    # onde cada probabilidade representa a chance de um determinado token do vocabulário ser o token mascarado.
    mlm_output = layers.Dense( config.VOCAB_SIZE, name = "mlm_cls", activation = "softmax" )(
            encoder_output
    )

    # Cria uma instância do modelo MaskedLanguageModel que definimos anteriormente.
    # As entradas do modelo são as sequências de tokens ('inputs') e a saída é a previsão dos tokens mascarados ('mlm_output').
    mlm_model = MaskedLanguageModel( inputs, mlm_output, name = "masked_bert_model" )

    # Define o otimizador a ser usado para o treinamento.
    # 'Adam' é um algoritmo de otimização popular. A taxa de aprendizado ('learning_rate') é definida em 'config.LR'.
    optimizer = keras.optimizers.Adam( learning_rate = config.LR )

    # Compila o modelo, configurando o otimizador. A função de perda é definida dentro da classe MaskedLanguageModel.
    mlm_model.compile( optimizer = optimizer )

    # Retorna a instância do modelo BERT para MLM.
    return mlm_model


# Cria um mapeamento do ID do token para o token em si.
# 'vectorize_layer.get_vocabulary()' retorna uma lista de todos os tokens no vocabulário.
# 'enumerate' cria pares de índice e token, e 'dict()' transforma esses pares em um dicionário.
id2token = dict( enumerate( vectorize_layer.get_vocabulary() ) )

# Cria um mapeamento do token para o seu ID.
# Este dicionário é criado invertendo o dicionário 'id2token'. Para cada valor (token) em 'id2token',
# sua chave (ID) se torna o novo valor, e o token se torna a nova chave.
token2id = { y: x for x, y in id2token.items() }

from tensorflow import keras
import numpy as np
from pprint import pprint
import torch  # Importa torch para verificar o tipo do tensor


# Por herdar de keras.callbacks.Callback, esta classe pode ser usada durante o treinamento
# para executar ações personalizadas ao final de cada época (ou em outros momentos do treinamento).
class MaskedTextGenerator( keras.callbacks.Callback ):
    def __init__( self, sample_tokens, top_k = 5 ):
        self.sample_tokens = sample_tokens  # Exemplo de entrada que contém o token de máscara para previsão.
        self.k = top_k  # Número de candidatos principais (tokens com maior probabilidade) a serem considerados para substituir a máscara.

    # Converte uma sequência de IDs de tokens para uma string legível.
    def decode( self, tokens ):
        return " ".join( [ id2token.get( int( t ), "[UNK]" ) for t in tokens if t != 0 ] )

    # Converte um ID de token para seu token correspondente.
    def convert_ids_to_tokens( self, id ):
        return id2token[ id ]

    # Este método é executado ao final de cada época de treinamento.
    def on_epoch_end( self, epoch, logs = None ):
        prediction = self.model.predict( self.sample_tokens )

        # Procura o índice onde o token de máscara está presente na sequência de tokens de entrada.
        # 'np.where' retorna um array de índices onde a condição é verdadeira.
        # Como 'self.sample_tokens' pode ser um lote (mesmo que de tamanho 1), pegamos o índice dentro da sequência.
        masked_index = np.where( self.sample_tokens == mask_token_id )
        # 'masked_index' será uma tupla de arrays (para cada dimensão). Pegamos os índices da segunda dimensão (a sequência).
        masked_index = masked_index[ 1 ]

        # 'prediction' terá a mesma forma que a entrada, exceto que a última dimensão conterá as probabilidades
        # para cada token do vocabulário. Pegamos as probabilidades de predição para o token mascarado na sequência.
        mask_prediction = prediction[ 0 ][ masked_index ]

        # Para o token mascarado, obtemos os índices dos 'top_k' tokens com as maiores probabilidades.
        # 'argsort()' retorna os índices que ordenariam um array. Usamos '[-self.k:]' para pegar os últimos 'k' índices (maiores probabilidades)
        # e '[::-1]' para inverter a ordem, obtendo os índices em ordem decrescente de probabilidade.
        top_indices = mask_prediction[ 0 ].argsort()[ -self.k: ][ ::-1 ]
        # Obtemos os valores das probabilidades correspondentes aos 'top_indices'.
        values = mask_prediction[ 0 ][ top_indices ]

        # Itera sobre os 'top_k' candidatos previstos.
        for i in range( len( top_indices ) ):
            # Obtém o índice do token previsto.
            p = top_indices[ i ]
            # Obtém a probabilidade correspondente.
            v = values[ i ]

            # Cria uma cópia da sequência de tokens de entrada para não modificar a original.
            # Verifica se self.sample_tokens é um tensor do PyTorch e o move para a CPU se necessário.
            if isinstance( self.sample_tokens, torch.Tensor ):
                tokens = np.copy( self.sample_tokens[ 0 ].cpu().numpy() )
            else:
                tokens = np.copy( self.sample_tokens[ 0 ] )

            # Substitui o token de máscara na cópia da sequência pelo token previsto atual.
            tokens[ masked_index[ 0 ] ] = p

            # Cria um dicionário contendo o texto original, a previsão com o token substituído,
            # a probabilidade da previsão e o token previsto.
            result = {
                "input_text": self.decode( self.sample_tokens[ 0 ].cpu().numpy() ) if isinstance(
                        self.sample_tokens,
                        torch.Tensor
                ) else self.decode(
                        self.sample_tokens[ 0 ]
                ),
                "prediction": self.decode( tokens ),
                "probability": v,
                "predicted mask token": self.convert_ids_to_tokens( p ),
            }
            # Imprime o resultado formatado para facilitar a leitura.
            pprint( result )


# Converte o exemplo de texto contendo a máscara para uma sequência de IDs de tokens
# utilizando a camada de vetorização ('vectorize_layer') que foi adaptada aos dados de treino.
sample_tokens = vectorize_layer( [ "Eu tenho [mask] " ] )

# Cria uma instância do callback MaskedTextGenerator, que será usado para gerar exemplos de predição
# ao final de cada época de treinamento.
# Verifica se 'sample_tokens' é um tensor do PyTorch. Se for, move-o para a CPU e o converte para um array NumPy
# antes de passá-lo para o callback, pois o callback espera um array NumPy ou similar.
if isinstance( sample_tokens, torch.Tensor ):
    generator_callback = MaskedTextGenerator(
            sample_tokens.cpu().numpy()
    )  # Movendo para a CPU e convertendo para numpy
else:
    generator_callback = MaskedTextGenerator( sample_tokens )

# Cria uma instância do modelo BERT para a tarefa de Masked Language Modeling (MLM)
# chamando a função 'create_masked_language_bert_model' que define a arquitetura do modelo.
bert_masked_model = create_masked_language_bert_model()

# Exibe um resumo da arquitetura do modelo BERT criado.
# O método 'summary()' mostra as camadas do modelo, suas formas de saída e o número de parâmetros treináveis e não treináveis.
# bert_masked_model.summary()

# Calcula o número de passos (batches) por época para os conjuntos de treino e validação.
# O número de passos é determinado dividindo o número total de amostras pelo tamanho do lote (BATCH_SIZE).
# Usamos len(train_df) para obter o número de amostras no DataFrame de treino.
train_steps_per_epoch: int = len( train_df ) // config.BATCH_SIZE
# Usamos len(test_df) para obter o número de amostras no DataFrame de teste/validação.
val_steps_per_epoch: int = len( test_df ) // config.BATCH_SIZE

# Imprime o número calculado de passos por época para treino e validação.
print( f"Passos por época (Treino): {train_steps_per_epoch}" )
print( f"Passos por época (Validação): {val_steps_per_epoch}" )

# Inicia o treinamento do modelo BERT para Masked Language Modeling (MLM).
# O método 'fit' é usado para treinar o modelo com os dados de treinamento.
history = bert_masked_model.fit(
        # Dataset de treinamento para MLM, contendo as entradas mascaradas, rótulos e pesos.
        mlm_train_ds,
        validation_data = mlm_val_ds,
        # Dataset de validação para MLM, usado para avaliar o desempenho do modelo durante o treinamento.
        epochs = 5,
        # Define o número total de épocas (passagens completas pelos dados de treinamento) para treinar o modelo.
        steps_per_epoch = train_steps_per_epoch,
        # Especifica quantos passos de otimização devem ser realizados em cada época para o conjunto de treinamento.
        validation_steps = val_steps_per_epoch,
        # Especifica quantos passos de avaliação devem ser realizados em cada época para o conjunto de validação.
        callbacks = [ generator_callback ]
        # Passa uma lista de callbacks para serem executados em diferentes estágios do treinamento.
        # Aqui, 'generator_callback' será chamado ao final de cada época para gerar exemplos de predição.
)

# Salva o modelo BERT treinado para a tarefa de Masked Language Modeling (MLM) em um arquivo.
# O formato ".keras" é o formato padrão para salvar modelos no Keras 3.
bert_masked_model.save( "bert_mlm.keras" )

# Imprime uma mensagem indicando que o treinamento foi concluído e o modelo foi salvo com sucesso.
print( "\nTreinamento concluído e modelo salvo!" )
