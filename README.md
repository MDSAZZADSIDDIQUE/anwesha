# Anwesha RAG Chatbot: A Deep Dive

Anwesha is a state-of-the-art Retrieval-Augmented Generation (RAG) chatbot meticulously engineered to navigate and answer questions from Rabindranath Tagore's classic short story, "Aparichita." This project stands as a testament to the power of modern AI in understanding and interacting with complex, multilingual literary texts, offering accurate, context-aware responses in both its original Bangla and English.

*(A high-level diagram illustrating the multi-query retrieval process)*

-----

## 🌟 About The Project

The primary motivation behind Anwesha was to tackle a significant challenge in the world of natural language processing: making literary works in non-English languages, specifically Bengali, accessible to AI-driven analysis and interaction. Standard PDF parsing tools often fail with the complexities of Bengali script and document layouts. This project was born from the desire to overcome these hurdles and build a highly accurate, robust, and user-friendly chatbot that could serve as a reliable companion for readers and researchers of Bengali literature.

### ✨ Highlighted Features

Anwesha is more than just a chatbot; it's a showcase of a sophisticated AI pipeline. Here’s a closer look at its core features:

  * [cite\_start]**State-of-the-Art Parsing:** The system integrates **LlamaParse** in `parse_document_with_lvm` mode, powered by `anthropic-sonnet-4.0`[cite: 4]. [cite\_start]This isn't just standard text extraction; it's an intelligent parsing that understands the semantic structure of the PDF, preserving layouts, tables, and lists, which is crucial for the integrity of the literary text[cite: 48, 49].
  * [cite\_start]**Relevance-Focused Reranking:** To go beyond simple similarity search, Anwesha implements a reranking mechanism using **reciprocal rank fusion**[cite: 103]. This ensures that the documents passed to the language model are not just similar but are the most relevant, leading to higher quality answers.
  * **Multi-Query Construction:** Ambiguity is a common problem in user queries. [cite\_start]Anwesha addresses this by using a language model to generate five different versions of a user's question[cite: 113]. [cite\_start]This query expansion technique casts a wider net, dramatically increasing the odds of retrieving the most relevant context, even if the user's original phrasing is vague[cite: 114, 116].
  * **Comprehensive Evaluation:** The system's performance isn't just a subjective claim. [cite\_start]It was rigorously evaluated using the **Ragas** library, a specialized framework for assessing RAG pipelines on metrics like Faithfulness, Context Recall, and Factual Correctness[cite: 144, 145].
  * [cite\_start]**Seamless Deployment:** Anwesha is brought to life using **Streamlit** for a user-friendly, interactive web interface, and was initially designed with a **Flask** backend, making it both powerful and accessible[cite: 9, 207, 208].

-----

## 🚀 The Vision: Future Enhancements

The current version of Anwesha is a robust and capable system, but the journey doesn't end here. Here are the exciting improvements planned for the future:

  * [cite\_start]**CRAG (Corrective Retrieval-Augmented Generation):** To ensure the chatbot provides the most current and factually accurate information, the plan is to integrate web search functionality[cite: 10]. This would allow the system to self-correct and augment its knowledge base with information from the internet.
  * [cite\_start]**Fine-Tuned Embedding Model:** While the current multilingual model performs exceptionally well, the next step is to **fine-tune a state-of-the-art model like `gemini-embedding-001`**[cite: 11, 147]. This would tailor the embedding space specifically to the nuances of Bengali literature and the project's domain, promising a significant boost in retrieval accuracy.
  * [cite\_start]**Hybrid Search:** To get the best of both worlds, the project aims to implement hybrid search[cite: 12]. This would combine traditional keyword-based search (like TF-IDF or BM25) with modern semantic search, ensuring that even queries with very specific, rare keywords are handled effectively.
  * **LLM-Powered Dynamic Cleaning:** Manual data cleaning can be a bottleneck. [cite\_start]The future vision includes a system where a language model can **dynamically and intelligently clean the extracted text**, removing artifacts and formatting inconsistencies on the fly without manual intervention[cite: 13, 52].
  * [cite\_start]**Advanced Indexing with RAPTOR and ColBERT:** To achieve a new level of precision, the plan is to implement cutting-edge indexing techniques like **RAPTOR** and **ColBERT**[cite: 148]. These methods create more sophisticated representations of the documents, leading to highly accurate and granular retrieval.
  * [cite\_start]**Adaptive RAG:** The ultimate goal is to evolve Anwesha into an **Adaptive RAG system**[cite: 149]. [cite\_start]This advanced strategy would allow the system to learn from its interactions, analyze incoming queries to determine their complexity, and dynamically choose the best retrieval strategy, effectively creating a self-improving and self-correcting AI[cite: 150].

-----

## 🛠️ Tech Stack & Architecture

Anwesha is built on a modern, powerful stack designed for building high-performance AI applications.

| Component | Technology | Role in Project |
| :--- | :--- | :--- |
| **Application Framework** | `Streamlit` | [cite\_start]Provides the interactive, user-friendly web interface for the chatbot[cite: 179]. |
| **Core AI Framework** | `LangChain` | [cite\_start]The backbone of the project, used to orchestrate the entire RAG pipeline, from data ingestion to generation[cite: 182]. |
| **LLM Inference** | `Groq` | [cite\_start]Provides access to blazing-fast LLaMA 3 inference, powering the generation and query transformation stages[cite: 169, 184]. |
| **PDF Parsing**| `llama-parse`| [cite\_start]The core tool for intelligently parsing the source PDF document, preserving its structure and content[cite: 189].|
| **Embedding Model**| `intfloat/multilingual-e5-large-instruct`| [cite\_start]A powerful multilingual model used to convert text into high-dimensional vectors for semantic comparison[cite: 83, 91].|
| **Vector Database** | `ChromaDB` | [cite\_start]A lightweight and efficient open-source vector store used to store and retrieve document embeddings[cite: 100, 105, 194]. |
| **Evaluation** | `Ragas` | [cite\_start]A specialized framework used to quantitatively measure the performance and reliability of the RAG pipeline[cite: 144, 201]. |

-----

## ⚙️ Setup and Local Installation Guide

Follow these steps to get a local copy of Anwesha up and running on your machine.

#### 1\. Prerequisites

Ensure you have **Python (version 3.8 or higher)** and **Git** installed on your system.

#### 2\. Clone the Repository

Open your terminal and clone the project repository.

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

#### 3\. Create and Activate a Virtual Environment

It is a best practice to use a virtual environment to isolate project dependencies and avoid conflicts with other Python projects.

```bash
# Create the virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 4\. Install Dependencies

Install all the necessary Python packages listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

#### 5\. Set Up Environment Variables

The application requires an API key from Groq to power its language model.

  * Create a file named `.env` in the root directory of the project.

  * Add your Groq API key to this file in the following format:

    ```
    GROQ_API_KEY="your_actual_groq_api_key_here"
    ```

#### 6\. Run the Application

The user interface is built with Streamlit. To start the application, run the following command in your terminal:

```bash
streamlit run app.py
```

This command will launch the application, and it should open automatically in your default web browser.

-----

## 🔧 A Deep Dive into the Technical Implementation

This section provides a detailed breakdown of the critical design choices and the reasoning behind them, showcasing the journey from problem to solution.

### 1\. The Challenge of Text Extraction

#### The Problem

[cite\_start]The first and most significant hurdle was extracting clean, structured text from a Bengali PDF[cite: 19]. [cite\_start]Most standard Python libraries for PDF parsing, such as `PyPDF`, `PyMuPDF`, and `PDFPlumber`, are optimized for Latin scripts and struggled immensely with Bengali characters, often rendering them as gibberish[cite: 20, 21, 23, 24, 25].

#### The Journey to a Solution

Several alternatives were explored in the quest for a reliable parsing solution:

  * [cite\_start]**Initial Success with OCR:** `UnstructuredPDFLoader`, when switched from its default `hi_res` strategy to `ocr_only`, provided a breakthrough[cite: 26, 29]. [cite\_start]By leveraging the Tesseract OCR engine in its backend, it could recognize the Bengali script far better than other libraries, though it still had minor issues[cite: 30, 43].
  * **Dead Ends:** Other modern tools were tested but proved unsuitable. [cite\_start]`Mistral-ocr` did not even support the Bengali language[cite: 32, 33]. [cite\_start]`Marker-pdf`, a promising tool, failed due to issues with downloading its layout models[cite: 34].
  * [cite\_start]**The "Holy Grail": `LlamaParse`:** The search concluded with the discovery of `LlamaParse`[cite: 44, 45]. This tool was a game-changer. [cite\_start]It was specifically engineered to handle complex, semi-structured documents, correctly identifying paragraphs, lists, and other elements with remarkable precision[cite: 47, 48]. [cite\_start]Its ability to preserve the semantic layout of the source material was critical for a literary work where structure carries meaning[cite: 49].

### 2\. The Strategy for Document Cleaning

Once the text was extracted, it needed to be cleaned of noise like headers, footers, and formatting artifacts. An initial, more ambitious approach involved using an LLM for dynamic cleaning.

```python
// This code demonstrates the dynamic cleaning approach
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000, chunk_overlap=100
)
splits = text_splitter.split_documents(docs)

// Setup the LLM chain for cleaning
llm = ChatGroq(model="moonshotai/kimi-k2-instruct")
chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Clean and delete unnecessary stuffs from the following document without changing the conetxt:\n\n{doc}")
    | llm
    | StrOutputParser()
)

// Batch process the documents for cleaning
cleaned_documents = chain.batch(splits, {"max_concurrency": 10})
```

[cite\_start]While this method was intelligent and effective, it was computationally expensive and quickly consumed the available Groq API limits[cite: 70]. [cite\_start]As a result, the project reverted to a more pragmatic approach of **manual cleaning using regular expressions (regex)** to fix specific, recurring issues[cite: 71].

### 3\. The Art of Chunking

[cite\_start]**Strategy:** The project employs a `RecursiveCharacterTextSplitter`[cite: 76].

**Reasoning:** How you split a document into chunks is paramount for semantic retrieval. [cite\_start]A naive strategy, like splitting every 1000 characters, could cut a sentence or a question-and-answer pair in half, destroying its meaning[cite: 79]. The recursive splitter is far more intelligent. It attempts to split along a hierarchy of separators, prioritizing logical breaks like paragraphs (`\n\n`) first, then sentences, and so on. [cite\_start]For a Q\&A document, this ensures that a question and its corresponding answer are highly likely to remain in the same chunk, providing the embedding model with rich, contextually complete data to work with[cite: 77, 78].

### 4\. Choosing the Right Embedding Model

[cite\_start]**Model Used:** `intfloat/multilingual-e5-large-instruct`[cite: 83].

**Reasoning:** The choice of embedding model was critical for the project's success. After extensive experimentation, this model was selected for three compelling reasons:

1.  [cite\_start]**True Multilingual Power:** It demonstrated excellent performance for both **Bangla** and **English**, a non-negotiable requirement for this bilingual application[cite: 92]. [cite\_start]A dedicated Bengali model from Hugging Face, surprisingly, performed terribly[cite: 86, 87].
2.  [cite\_start]**Instruction-Tuned for Retrieval:** The `-instruct` variant is specifically fine-tuned to understand the *intent* behind a user's query[cite: 93]. This makes it far more effective for retrieval tasks than general-purpose models, as it's trained to produce embeddings that are optimized for similarity search based on a query.
3.  [cite\_start]**Top-Tier Performance:** This model is a recognized top performer on the **MTEB (Massive Text Embedding Benchmark)**, providing confidence in its ability to generate high-quality and reliable vector representations of the text[cite: 94].

### 5\. Handling Vague Queries with Multi-Query Retrieval

**The Problem:** Users often ask questions that are ambiguous or use different terminology than the source document. A direct similarity search might fail in these cases.

[cite\_start]**The Solution:** Anwesha implements a **multi-query retrieval** strategy[cite: 112]. [cite\_start]Instead of using the user's query directly, it first passes the query to an LLM, which generates five distinct but semantically similar versions of the question[cite: 113, 116].

```python
// This code generates multiple query perspectives
template = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives
    | ChatGroq(model="moonshotai/kimi-k2-instruct")
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)
```

Each of these five queries is then used to retrieve documents from the vector store. [cite\_start]This approach significantly broadens the search space, increasing the likelihood of finding relevant context and making the system far more robust and forgiving of vague user input[cite: 114]. [cite\_start]This technique was chosen after experimenting with other methods like Rag-Fusion, Decomposition, and Step-back, as it yielded the best results for this specific use case[cite: 115, 133, 134, 135].

-----

## 📊 A Note on Evaluation

The quality of Anwesha is not an afterthought; it was a core part of the development process. The system was rigorously evaluated using the `ragas` library, focusing on three key metrics:

  * [cite\_start]**Faithfulness:** This measures whether the generated answer is factually grounded in the retrieved context[cite: 216]. [cite\_start]A high score means the model is not "hallucinating" or inventing information[cite: 217].
  * [cite\_start]**Context Recall:** This metric assesses whether the retrieval system successfully found all the necessary information from the source text required to comprehensively answer the question[cite: 218].
  * [cite\_start]**Factual Correctness:** This compares the generated answer against a ground-truth (a manually verified correct answer) to determine its factual accuracy[cite: 219].

[cite\_start]**Important:** The evaluation code was **intentionally removed** from the final `app.py` script for deployment[cite: 220]. [cite\_start]This was a deliberate choice to create a lighter, faster application with fewer dependencies, focusing the user experience solely on the chatbot's functionality[cite: 221, 222, 224]. [cite\_start]The detailed evaluation code and its results are preserved in the development notebooks (`anwesha_version_0_final_version.ipynb`) in the project repository for reference[cite: 225].

-----