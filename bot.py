import streamlit as st
from pypdf import PdfReader

# LangChain e Hugging Face
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

# Per caricare variabili d'ambiente (token, ecc.)
from dotenv import load_dotenv
load_dotenv()


def read_pdf(file):
    """Legge un file PDF e restituisce l'intero contenuto testuale."""
    document = ""
    reader = PdfReader(file)
    for page in reader.pages:
        document += page.extract_text()
    return document


def read_txt(file):
    """Legge un file di testo e converte i caratteri di newline."""
    document = str(file.getvalue())
    document = document.replace("\\n", " \\n ").replace("\\r", " \\r ")
    return document


def split_doc(document, chunk_size, chunk_overlap):
    """Suddivide il testo in chunk più piccoli per l'embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split = splitter.split_text(document)
    split = splitter.create_documents(split)
    return split


def embedding_storing(split, create_new_vs, existing_vector_store, new_vs_name):
    """
    Crea o aggiorna un VectorStore FAISS con i documenti embeddati
    usando un modello di embedding (all-MiniLM-L6-v2).
    """
    if create_new_vs is not None:
        instructor_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.from_documents(split, instructor_embeddings)

        if create_new_vs is True:
            # Crea un nuovo vector store
            db.save_local("vectorstore/" + new_vs_name)
        else:
            # Carica il vector store esistente e unisci i dati
            load_db = FAISS.load_local(
                "vectorstore/" + existing_vector_store,
                instructor_embeddings,
                allow_dangerous_deserialization=True
            )
            load_db.merge_from(db)
            load_db.save_local("vectorstore/" + new_vs_name)

        st.success("The document has been saved.")


#
# PROMPT PERSONALIZZATO
#
custom_prompt_template = """
Sei un assistente AI estremamente competente. Rispondi in modo completo, ricco di dettagli e chiaro,
tenendo conto delle informazioni fornite nei documenti (context) e della conversazione fin qui.

- Se non hai abbastanza informazioni, dillo esplicitamente.
- Fornisci esempi, spiegazioni aggiuntive o suggerimenti correlati se rilevanti.
- Usa uno stile discorsivo e amichevole, ma professionale.

CONVERSAZIONE FINO AD ORA:
{chat_history}

DOMANDA DELL'UTENTE:
{question}

CONTESTO (frammenti dai documenti):
{context}

RISPOSTA COMPLETA:
"""
QA_PROMPT = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["chat_history", "question", "context"]
)


def prepare_rag_llm(token, vector_store_list, temperature, max_length):
    """
    Prepara la chain di Retrieval-Augmented Generation caricando il vector store
    e il modello Falcon 7B Instruct. Usa un prompt personalizzato per risposte più ricche.
    """
    # Carica embeddings
    instructor_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Carica VectorStore
    loaded_db = FAISS.load_local(
        f"vectorstore/{vector_store_list}",
        instructor_embeddings,
        allow_dangerous_deserialization=True
    )

    # Configura l'LLM
    llm = HuggingFaceHub(
        repo_id='tiiuae/falcon-7b-instruct',
        huggingfacehub_api_token=token,
        model_kwargs={
            "temperature": temperature,  # Più alto -> risposte più creative
            "max_length": max_length,    # Numero di token massimo
            "top_p": 0.9                 # Ulteriore controllo sulla diversità dell'output
        }
    )

    # Memoria conversazionale
    memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    # Crea la chain di conversazione con retrieval
    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=loaded_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        memory=memory,
        # Prompt personalizzato
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )

    return qa_conversation


def generate_answer(question, token):
    """
    Utilizza la chain in session_state per generare la risposta,
    senza troncare il testo con lo split su "Helpful Answer:".
    """
    if token == "":
        return "Insert the Hugging Face token", ["no source"]

    # Chiamata alla chain
    response = st.session_state.conversation({"question": question})

    # Ricava la risposta e i documenti di origine
    answer = response.get("answer", "")
    explanation = response.get("source_documents", [])
    doc_source = [d.page_content for d in explanation]

    return answer, doc_source

