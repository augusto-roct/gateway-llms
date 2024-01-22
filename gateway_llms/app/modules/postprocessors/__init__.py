from llama_index import postprocessor


def get_similarity(config: dict):
    return postprocessor.SimilarityPostprocessor(
        similarity_cutoff=config["similarity_cutoff"]
    )


def get_keyword(config: dict):
    return postprocessor.KeywordNodePostprocessor(
        required_keywords=config["required_keywords"],
        exclude_keywords=config["exclude_keywords"],
        lang=config["lang"]
    )


def get_metadata_replacement(config: dict):
    return postprocessor.MetadataReplacementPostProcessor(
        target_metadata_key=config["target_metadata_key"],
    )


def get_log_context_reorder():
    return postprocessor.LongContextReorder()


def get_sentence_embedding_optimizer(config: dict):
    return postprocessor.SentenceEmbeddingOptimizer(
        embed_model=config["service_context"].embed_model,
        percentile_cutoff=config["percentile_cutoff"],
        threshold_cutoff=config["threshold_cutoff"]
    )


def get_sentence_transformer_rerank(config: dict):
    return postprocessor.SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        top_n=config["top_n"]
    )


def get_llm_rerank(config: dict):
    return postprocessor.LLMRerank(
        top_n=config["top_n"],
        service_context=config["service_context"]
    )


def get_fix_recency(config: dict):
    return postprocessor.FixedRecencyPostprocessor(
        tok_k=config["tok_k"],
        date_key="date",
        service_context=config["service_context"]
    )


def get_embedding_recency(config: dict):
    return postprocessor.EmbeddingRecencyPostprocessor(
        similarity_cutoff=config["similarity_cutoff"],
        date_key="date",
        service_context=config["service_context"]
    )


def get_time_weight(config: dict):
    return postprocessor.TimeWeightedPostprocessor(
        time_decay=config["time_decay"],
        top_k=config["top_k"]
    )


def get_prev_next_node(config: dict):
    return postprocessor.PrevNextNodePostprocessor(
        docstore=config["index"].docstore,
        num_nodes=config["num_nodes"],
        mode=config["mode"]
    )


def get_auto_prev_next_node(config: dict):
    return postprocessor.AutoPrevNextNodePostprocessor(
        docstore=config["index"].docstore,
        num_nodes=config["num_nodes"],
        service_context=config["service_context"]
    )


def get_rank_gpt_rerank(config: dict):
    return postprocessor.RankGPTRerank(
        top_n=config["top_n"],
        llm=config["llm"]
    )


__all__ = [
    "get_similarity",
    "get_Keyword",
    "get_metada_replacement",
    "get_log_context_reorder",
    "get_sentence_embedding_optimizer",
    "get_sentence_transformer_rerank",
    "get_llm_rerank",
    "get_fix_recency",
    "get_embedding_recency",
    "get_time_weight",
    "get_prev_next_node",
    "get_auto_prev_next_node",
    "get_rank_gpt_rerank"
]
