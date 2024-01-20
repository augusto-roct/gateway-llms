from llama_index import LLMPredictor
from llama_index.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    KeywordExtractor,
    SummaryExtractor
)


def get_title(llm: LLMPredictor, nodes_quantity: int):
    title_extractor = TitleExtractor(
        llm=llm, nodes=nodes_quantity, num_workers=30)

    return title_extractor


def get_questions_answered(llm: LLMPredictor, questions_quantity: int):
    qa_extractor = QuestionsAnsweredExtractor(
        questions=questions_quantity, llm=llm, num_workers=30)

    return qa_extractor


def get_keywords(llm: LLMPredictor, quantity_keywords: int):
    keyword_extractor = KeywordExtractor(
        llm=llm, keywords=quantity_keywords, num_workers=30)

    return keyword_extractor


def get_summary(llm: LLMPredictor, types_summaries: list[str]):
    summary_extractor = SummaryExtractor(
        llm=llm, summaries=types_summaries, num_workers=30)

    return summary_extractor


__all__ = [
    "get_title",
    "get_questions_answered",
    "get_keywords",
    "get_summary"
]
