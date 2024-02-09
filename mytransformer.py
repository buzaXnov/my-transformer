import numpy as np

d_embedding = 4
d_key = d_value = d_query = 4
d_feed_forward = 8
n_attention_heads = 2
epsilon = 1e-6
gamma = 1
beta = 0


def encoder_decoder_attention(
    attention_input: np.array,
    encoder_output: np.array,
    WQ: np.array,
    WK: np.array,
    WV: np.array,
) -> np.array:
    """
    attention_input: the output of the previous attention!
    """
    K = encoder_output @ WK
    V = encoder_output @ WV
    Q = attention_input @ WQ

    return softmax((Q @ K.T) / np.sqrt(K.shape[1])) @ V


def multihead_encoder_decoder_attention(
    encoder_output: np.array,
    attention_input: np.array,
    WQs: np.array,
    WKs: np.array,
    WVs: np.array,
) -> np.array:
    attentions = np.concatenate(
        [
            encoder_decoder_attention(attention_input, encoder_output, WQ, WK, WV)
            for WQ, WK, WV in zip(WQs, WKs, WVs)
        ],
        axis=1,
    )

    W = np.random.randn(n_attention_heads * d_value, d_embedding)
    return attentions @ W


def decoder_block(
    x,
    encoder_output,
    WQs_self_attention,
    WKs_self_attention,
    WVs_self_attention,
    WQs_ed_attention,
    WKs_ed_attention,
    WVs_ed_attention,
    W1,
    b1,
    W2,
    b2,
) -> np.array:
    Z = multihead_attention(
        x, WQs_self_attention, WKs_self_attention, WVs_self_attention
    )
    Z = layer_norm(x + Z)  # LayerNorm of the Residual on attention output

    enc_dec_attn = multihead_encoder_decoder_attention(
        encoder_output, Z, WQs_ed_attention, WKs_ed_attention, WVs_ed_attention
    )
    enc_dec_attn = layer_norm(enc_dec_attn + Z)

    output = feed_forward(enc_dec_attn, W1, W2, b1, b2)

    return layer_norm(output + enc_dec_attn)  # LayerNorm of the Residual on FFN output


def random_decoder_block(x: np.array, encoder_output: np.array) -> np.array:
    WQs_self_attention = [
        np.random.randn(d_embedding, d_query) for _ in range(n_attention_heads)
    ]
    WKs_self_attention = [
        np.random.randn(d_embedding, d_key) for _ in range(n_attention_heads)
    ]
    WVs_self_attention = [
        np.random.randn(d_embedding, d_value) for _ in range(n_attention_heads)
    ]

    WQs_ed_attention = [
        np.random.randn(d_embedding, d_query) for _ in range(n_attention_heads)
    ]
    WKs_ed_attention = [
        np.random.randn(d_embedding, d_key) for _ in range(n_attention_heads)
    ]
    WVs_ed_attention = [
        np.random.randn(d_embedding, d_value) for _ in range(n_attention_heads)
    ]

    W1 = np.random.randn(d_embedding, d_feed_forward)
    b1 = np.random.randn(d_feed_forward)
    W2 = np.random.randn(d_feed_forward, d_embedding)
    b2 = np.random.randn(d_embedding)

    return decoder_block(
        x,
        encoder_output,
        WQs_self_attention,
        WKs_self_attention,
        WVs_self_attention,
        WQs_ed_attention,
        WKs_ed_attention,
        WVs_ed_attention,
        W1,
        b1,
        W2,
        b2,
    )


def decoder(x, decoder_embedding, n=6) -> np.array:
    for _ in range(n):
        x = random_decoder_block(x, decoder_embedding)
    return x


def softmax(mat: np.array) -> np.array:
    return np.exp(mat) / np.sum(a=np.exp(mat), axis=1, keepdims=True)


def attention(E: np.array, WQ: np.array, WK: np.array, WV: np.array) -> np.array:
    Q = E @ WQ
    K = E @ WK
    V = E @ WV

    return softmax((Q @ K.T) / np.sqrt(K.shape[1])) @ V


def multihead_attention(x, WQs, WKs, WVs) -> np.array:
    attentions = np.concatenate(
        [attention(x, WQ, WK, WV) for WQ, WK, WV in zip(WQs, WKs, WVs)], axis=1
    )
    W = np.random.randn(n_attention_heads * d_value, d_embedding)
    return attentions @ W


def relu(x: np.array) -> np.array:
    return np.maximum(0, x)


def feed_forward(
    x: np.array, W1: np.array, W2: np.array, b1: np.array, b2: np.array
) -> np.array:
    return relu(x @ W1 + b1) @ W2 + b2


def layer_norm(x: np.array, epsilon: float = 1e-6) -> np.array:
    mean = x.mean(axis=1, keepdims=True)
    # var = x.var(axis=1, keepdims=True)  # NOTE: variance is the standard deviation squared
    # To lower the amount of calcuations (avoid calc. var and then sqrt in the denominator) we calculate the std dev
    std_dev = x.std(axis=1, keepdims=True)
    return (x - mean) / std_dev + epsilon * gamma + beta


# New definition with layer normalization and residual connections
def encoder_block(
    x: np.array,
    WQs: np.array,
    WKs: np.array,
    WVs: np.array,
    W1: np.array,
    W2: np.array,
    b1: np.array,
    b2: np.array,
) -> np.array:
    Z = multihead_attention(x, WQs, WKs, WVs)
    Z = layer_norm(x + Z)  # LayerNorm of the Residual on attention output
    output = feed_forward(Z, W1, W2, b1, b2)

    return layer_norm(Z + output)  # LayerNorm of the Residual on FFN output


def random_encoder_block(x: np.array) -> np.array:
    WQs = [np.random.randn(d_embedding, d_query) for _ in range(n_attention_heads)]
    WKs = [np.random.randn(d_embedding, d_key) for _ in range(n_attention_heads)]
    WVs = [np.random.randn(d_embedding, d_value) for _ in range(n_attention_heads)]

    W1 = np.random.randn(d_embedding, d_feed_forward)
    b1 = np.random.randn(d_feed_forward)
    W2 = np.random.randn(d_feed_forward, d_embedding)
    b2 = np.random.randn(d_embedding)

    return encoder_block(x, WQs, WKs, WVs, W1, W2, b1, b2)


def encoder(x, n=6):
    for _ in range(n):
        x = random_encoder_block(x)
    return x
