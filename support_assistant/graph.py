import os

from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from .retrieval import retrieve


class GraphState(TypedDict):

    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


def classify_intent(
    state: GraphState
):

    query = state["query"].lower()

    is_policy = any(
        keyword in query
        for keyword in POLICY_KEYWORDS
    )

    if is_policy:

        intent = "policy_question"

    else:

        intent = "general_question"

    return {
        "intent": intent
    }


def retrieve_and_answer(
    state: GraphState
):

    ids, documents = retrieve(
        state["query"],
        top_k=3
    )

    top_chunk = documents[0]

    answer = (
        "Based on the retrieved context: "
        + top_chunk[:200]
    )

    return {
        "answer": answer,
        "sources": ids,
        "confidence": 1.0
    }


def direct_answer(
    state: GraphState
):

    return {
        "answer":
            "I can only answer questions about Zepto policies right now.",
        "sources": [],
        "confidence": 1.0
    }


def route(state):

    if state["intent"] == "policy_question":

        return "retrieve"

    return "direct"


graph_builder = StateGraph(
    GraphState
)


graph_builder.add_node(
    "classify_intent",
    classify_intent
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph_builder.add_node(
    "direct_answer",
    direct_answer
)


graph_builder.add_edge(
    START,
    "classify_intent"
)


graph_builder.add_conditional_edges(
    "classify_intent",
    route,
    {
        "retrieve":
            "retrieve_and_answer",

        "direct":
            "direct_answer"
    }
)


graph_builder.add_edge(
    "retrieve_and_answer",
    END
)

graph_builder.add_edge(
    "direct_answer",
    END
)


graph = graph_builder.compile()