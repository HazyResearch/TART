from .embed_layers import (
    LOOEmbeddingCausalLM,
    VanillaEmbeddingCausalLM,
    StreamEmbeddingCausalLM,
    StreamEmbeddingWhisper,
    StreamEmbeddingViT,
)

from .tart_datasets import (
    HateSpeech,
    SpeechCommands,
    SMSSpam,
    MNIST,
    AGNews,
    DBPedia14,
    CIFAR10,
    YelpPolarity,
)


# TODO: Avanika -- Restructure to not need a dict per model
EMBEDDING_REGISTRY_AC = {
    "gpt-neo-125m": {
        "stream": StreamEmbeddingCausalLM,
        "loo": LOOEmbeddingCausalLM,
        "vanilla": VanillaEmbeddingCausalLM,
    },
    "bloom-560m": {
        "stream": StreamEmbeddingCausalLM,
        "loo": LOOEmbeddingCausalLM,
        "vanilla": VanillaEmbeddingCausalLM,
    },
    "vit-base-patch16-224-in21k": {"stream": StreamEmbeddingViT},
    "vit-large-patch16-224-in21k": {"stream": StreamEmbeddingViT},
    "whisper-large": {"stream": StreamEmbeddingWhisper},
    "whisper-base": {"stream": StreamEmbeddingWhisper},
    "whisper-small": {"stream": StreamEmbeddingWhisper},
}


DOMAIN_REGISTRY = {"supported": ["text", "audio", "image"]}


DATASET_REGISTRY = {
    "text": {
        "hate_speech18": HateSpeech,
        "sms_spam": SMSSpam,
        "ag_news": AGNews,
        "dbpedia_14": DBPedia14,
        "yelp_polarity": YelpPolarity,
    },
    "audio": {"speech_commands": SpeechCommands},
    "image": {
        "mnist": MNIST,
        "cifar10": CIFAR10,
    },
}
