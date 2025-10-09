class model_config:
    hidden_size: int = 64
    intermediate_size: int = 256
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    block_size: int = 512
    max_position_embeddings: int = 2048
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = True
