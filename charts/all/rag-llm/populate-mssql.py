import os
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_sqlserver import SQLServer_VectorStore
from langchain_huggingface import HuggingFaceEmbeddings



doc_folder = os.getenv('DOC_LOCATION')
temp_folder = os.getenv('TEMP_DIR')

pdf_folder_path = temp_folder+'/source_repo/'+doc_folder
print('PDF folder:',pdf_folder_path)
loader = PyPDFDirectoryLoader(pdf_folder_path)
docs = loader.load()

vector_store = SQLServer_VectorStore(
    connection_string=os.getenv('MSSQL_URL'),
    #distance_strategy=DistanceStrategy.COSINE,
    embedding_function=HuggingFaceEmbeddings(),
    # embedding_length=1536,
    table_name="whatever",
)

# #### Split documents into chunks with some overlap
print(">>>>Document splitting .....")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=40)
all_splits = text_splitter.split_documents(docs)

print(">>>>Creating index .....")
# #### Create the index and ingest the documents



for i, doc in enumerate(all_splits):
    vector_store.add_documents(documents=[doc], ids=[f"doc_{i}"])


# ## Ingesting new documents
# #### Example with Web pages

loader = WebBaseLoader(["https://ai-on-openshift.io/getting-started/openshift/",
                        "https://ai-on-openshift.io/getting-started/opendatahub/",
                        "https://ai-on-openshift.io/getting-started/openshift-ai/",
                        "https://ai-on-openshift.io/odh-rhoai/configuration/",
                        "https://ai-on-openshift.io/odh-rhoai/custom-notebooks/",
                        "https://ai-on-openshift.io/odh-rhoai/nvidia-gpus/",
                        "https://ai-on-openshift.io/odh-rhoai/custom-runtime-triton/",
                        "https://ai-on-openshift.io/odh-rhoai/openshift-group-management/",
                        "https://ai-on-openshift.io/tools-and-applications/minio/minio/"
                       ])


print(">>>>Loading Documents .....")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=40)
all_splits = text_splitter.split_documents(data)
print(">>>Adding new documents from Web... ")

for i, doc in enumerate(all_splits):
    vector_store.add_documents(documents=[doc], ids=[f"doc_{i}"])
