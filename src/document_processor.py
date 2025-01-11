import os
import shutil

import operator
from typing import Annotated, List, Literal, TypedDict

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langgraph.constants import Send

from langgraph.graph import END, START, StateGraph

import asyncio
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_text_splitters import CharacterTextSplitter

async def process_document(temp_file_path):

    loader = Docx2txtLoader(temp_file_path)
    docs = loader.load()

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(temperature=0, model_name="gemma2-9b-it")
        # llm = ChatOllama(temperature=0, model="llama3")

    map_prompt = ChatPromptTemplate.from_messages(
            [("system", "Haz un resumen en español de lo soguiente como si fueras a exponerlo en una conferencia :\\n\\n{context}")]
        )

    reduce_template = """
    Lo siguiente es un conjunto de resúmenes:
    {docs}
    Toma estos y destílalos en un resumen final consolidado
    de los temas principales.
    """


    reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

        

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)
        

    token_max = 1000


    def length_function(documents: List[Document]) -> int:
        """Get number of tokens for input contents."""
        return sum(llm.get_num_tokens(doc.page_content) for doc in documents)


        # This will be the overall state of the main graph.
        # It will contain the input document contents, corresponding
        # summaries, and a final summary.
    class OverallState(TypedDict):
            # Notice here we use the operator.add
            # This is because we want combine all the summaries we generate
            # from individual nodes back into one list - this is essentially
            # the "reduce" part
        contents: List[str]
        summaries: Annotated[list, operator.add]
        collapsed_summaries: List[Document]
        final_summary: str


        # This will be the state of the node that we will "map" all
        # documents to in order to generate summaries
    class SummaryState(TypedDict):
        content: str


        # Here we generate a summary, given a document
    async def generate_summary(state: SummaryState):
        prompt = map_prompt.invoke(state["content"])
        response = await llm.ainvoke(prompt)
        return {"summaries": [response.content]}


        # Here we define the logic to map out over the documents
        # We will use this an edge in the graph
    def map_summaries(state: OverallState):
            # We will return a list of `Send` objects
            # Each `Send` object consists of the name of a node in the graph
            # as well as the state to send to that node
        return [
            Send("generate_summary", {"content": content}) for content in state["contents"]
        ]


    def collect_summaries(state: OverallState):
        return {
            "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
        }


    async def _reduce(input: dict) -> str:
        prompt = reduce_prompt.invoke(input)
        response = await llm.ainvoke(prompt)
        return response.content


        # Add node to collapse summaries
    async def collapse_summaries(state: OverallState):
        doc_lists = split_list_of_docs(
            state["collapsed_summaries"], length_function, token_max
        )
        results = []
        for doc_list in doc_lists:
            results.append(await acollapse_docs(doc_list, _reduce))

        return {"collapsed_summaries": results}


        # This represents a conditional edge in the graph that determines
        # if we should collapse the summaries or not
    def should_collapse(
        state: OverallState,
    ) -> Literal["collapse_summaries", "generate_final_summary"]:
        num_tokens = length_function(state["collapsed_summaries"])
        if num_tokens > token_max:
            return "collapse_summaries"
        else:
            return "generate_final_summary"


        # Here we will generate the final summary
    async def generate_final_summary(state: OverallState):
        response = await _reduce(state["collapsed_summaries"])
        return {"final_summary": response}


        # Construct the graph
        # Nodes:
    graph = StateGraph(OverallState)
    graph.add_node("generate_summary", generate_summary)  # same as before
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("collapse_summaries", collapse_summaries)
    graph.add_node("generate_final_summary", generate_final_summary)

        # Edges:
    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", END)

    app = graph.compile()
        
    async for step in app.astream(
        {"contents": [doc.page_content for doc in split_docs]},
        {"recursion_limit": 100},
    ):
        print(list(step.keys()))
            
    return step
                

  
    
    