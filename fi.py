from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

data_path = "data/"
db_stor = "vectors/db"

def crvector():
    load = DirectoryLoader(data_path,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    dock = load.load()
    textspl = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = textspl.split_documents(dock)
    emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(texts, emb)
    db.save_local(db_stor)
if __name__ == "__main__":
    crvector()