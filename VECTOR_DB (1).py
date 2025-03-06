import PyPDF2
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] ="sk-proj-cFrNlGjya1dSnksPjKXNT3BlbkFJxnT4jJWRjcV719eVwT5b"

def read_and_textify(files):
    text_list = []
    sources_list = [] # Initialize the sources_list variable
    for file in files:
        with open(file.name, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                text_list.append(text)
                sources_list.append(file.name + "_page_" + str(i))
    return [text_list, sources_list]

directory = r"C:\Users\DELL\Downloads\Resumes001 1 (1)\Resumes001\Backend_Resume"
files = os.listdir(directory)
files = [open(os.path.join(directory,x),"rb") for x in files if x.endswith(".pdf")]
print(files)              

textify_output = read_and_textify(files)

documents = textify_output[0]
sources = textify_output[1]

print(documents)

print(sources)

persist_directory = r"C:\Users\DELL\Downloads\Resumes001 1 (1)\Resumes001\Backend_Resume"
embeddings = OpenAIEmbeddings(openai_api_key =os.environ["OPENAI_API_KEY"])

vectordb = Chroma.from_texts(documents, embeddings,metadatas=[{"source": s} for s in sources],persist_directory=persist_directory)
model_name = "gpt-3.5-turbo"

vectordb.persist()
vectordb = None

vectordb = Chroma(persist_directory=persist_directory,
embedding_function=embeddings)
vectordb.get()

qa = VectorDBQAWithSourcesChain.from_chain_type(llm=OpenAI(), k=1,chain_type="stuff", vectorstore=vectordb)

query = "Print the skills of adilakshmi"
print(qa(query))
