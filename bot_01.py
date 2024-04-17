from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

db_path="vectors/db"
cprompt="""Please use the information provided to answer the user's question. If you're not sure of the answer, it's better to say you don't know than to guess.

Context: {context}
Question: {question}

Please only provide the answer that directly addresses the question. Avoid adding extra details.

Helpful Answer:"""
def custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=cprompt,
                            input_variables=['context', 'question'])
    return prompt

def retriev_qa(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain
#Loading the model
def load_model():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm
#QA Model Function
# Loading the FAISS vector store with dangerous deserialization enabled
def qbot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    # Set allow_dangerous_deserialization to True
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    llm = load_model()
    qa_prompt = custom_prompt()
    qa = retriev_qa(llm, qa_prompt, db)

    return qa

#output function
def result(query):
    qa_result = qbot()
    response = qa_result({'query': query})
    return response
#chainlit code
@cl.on_chat_start
async def start():
    chain = qbot()
    msg = cl.Message(content="Getting it ready.....")
    await msg.send()
    msg.content = "Halo. Welcome to MediAid. how can i help?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:"
    else:
        answer += "\n sources "

    await cl.Message(content=answer).send()