from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer

from config import TokenizerConfig

def build_hf_tokenizer() -> Tokenizer:
    """
    Build a fast HF `tokenizers` encoder from the trained vocab.json/merges.txt.

    Returns:
        Tokenizer: A ready-to-use fast BPE encoder.
    """
    model = BPE.from_file(str(TokenizerConfig.vocab_path), str(TokenizerConfig.merges_path))
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.add_special_tokens(list(TokenizerConfig.special_tokens))
    return tokenizer
