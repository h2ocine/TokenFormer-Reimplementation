class ModelConfig:
    def __init__(self, vocab_size, hidden_dim, num_heads, num_layers, num_tokens, max_seq_len):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
