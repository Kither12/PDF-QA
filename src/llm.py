import os

from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.vector_database import VectorDatabase

from langchain_cohere import ChatCohere, CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Try to keep the answer concise."
    "\n\n"
    "{context}"
)

system_parapharse = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \

Perform query expansion. If there are multiple common ways of phrasing a user question \
or common synonyms for key words in the question, make sure to return multiple versions \
of the query with the different phrasings.

If there are acronyms or words you are not familiar with, do not try to rephrase them.


{question}
"""


class LLM:
    def __init__(self, vector_database: VectorDatabase):
        self.llm = ChatCohere(model="command-r")

        compressor = CohereRerank(model="rerank-multilingual-v2.0")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vector_database.vector_store.as_retriever(),
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(
            compression_retriever,
            question_answer_chain,
        )

    def paraphrased_query(self, input: str) -> str:
        prompt = PromptTemplate.from_template(system_parapharse)
        a = self.llm.invoke(prompt.invoke({"question": input}))
        return str(a.content)

    def query(self, input: str) -> str:
        return self.rag_chain.invoke({"input": self.paraphrased_query(input)})["answer"]
