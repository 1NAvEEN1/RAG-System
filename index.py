from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from configs.config import settings

# Initialize FastAPI
app = FastAPI()

# Initialize API key and LLM
api_key = settings.OPENAI_API_KEY
llm = ChatOpenAI(model=settings.MODEL_NAME, streaming=True)
embedding_model = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)

# Document Loader
loader = PyPDFLoader(settings.PDF_PATH)
docs = loader.load()

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# Vector Store
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)

# Retriever
retriever = vectorstore.as_retriever()

# System Prompt
system_prompt = (
    "You are an intelligent chatbot. Use the following context to retrieve the content related to the given "
    "problem from the syllabus\n\n{context}"
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# QA Chain
qa_chain = create_stuff_documents_chain(llm, prompt)

# RAG Chain
rag_chain = create_retrieval_chain(retriever, qa_chain)


# Define SMEDetails model
class SMEInput(BaseModel):
    industry: str
    size: int
    digital_assets: str
    cybersecurity_maturity: str


@app.post("/get-recommendations/")
async def get_recommendations(sme_details: SMEInput):
    """
    Endpoint to get cybersecurity recommendations based on SME details.
    """
    # Format the input query for the RAG system
    input_query = (
        f"Provide cybersecurity recommendations for an SME operating in the {sme_details.industry} "
        f"sector with {sme_details.size} employees, handling {sme_details.digital_assets}, "
        f"and having {sme_details.cybersecurity_maturity} maturity in cybersecurity."
    )

    # Get response from RAG chain
    try:
        response = rag_chain.invoke({"input": input_query})
        return {"recommendations": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
