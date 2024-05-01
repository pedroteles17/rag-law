from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import dotenv
import os

dotenv.load_dotenv()

class VectorDB:
    def __init__(self, index_name):
        self.index_name = index_name
        self._verify_pinecone_key()

    def initialize_index(self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return PineconeVectorStore(index_name=self.index_name, embedding=embeddings)
    
    def _verify_pinecone_key(self):
        if os.getenv("PINECONE_API_KEY") is None:
            raise ValueError("PINECONE_API_KEY is not set")
    
class QueryEngine:
    SYSTEM_TEMPLATE = (
        "Como um sistema de perguntas e respostas inestimável para jurisprudência de Direito,"
        "sua função é fornecer respostas precisas e relevantes às consultas com base no contexto fornecido."
        "É importante confiar exclusivamente nas informações fornecidas e abster-se de oferecer respostas"
        "baseadas em conhecimentos externos. Se você não tem certeza sobre a resposta, é melhor não responder."
        "Não crie novas jurisprudências ou responda com conhecimentos externos. Sua tarefa é sugerir até"
        "três jurisprudências válidas para cada pergunta com base nas jurisprudências apresentadas"
        "no contexto. Explique o por quê de cada jurisprudência ser relevante para a pergunta."
        "Você deve incluir o 'nome_arquivo_pdf' das jurisprudências. Você receberá primeiro"
        "o contexto e, em seguida, a pergunta. Por favor, forneça a resposta com base no contexto fornecido."

        "<context>"
        "{context}"
        "</context>"

        "Pergunta: {input}"
    )
        
    def __init__(self, vector_store, similarity_top_k):
        self.vector_store = vector_store
        self.similarity_top_k = similarity_top_k
        self._verify_openai_key()

    def initialize_query_engine(self):
        return create_retrieval_chain(
            self.vector_store.as_retriever(),
            self.create_document_chain()
        )
    
    def create_document_chain(self):
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.0)
        prompt = ChatPromptTemplate.from_template(self.SYSTEM_TEMPLATE)
        return create_stuff_documents_chain(llm, prompt)

    def _verify_openai_key(self):
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OPENAI_API_KEY is not set")