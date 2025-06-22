# Smart Recruiter Assistant

A **Retrieval-Augmented Generation (RAG)** application that helps recruiters make informed hiring decisions by **analyzing candidate CVs** and **answering custom questions** using advanced NLP and LLMs.

Certainly — here is your structured and clean **Table of Contents** in plain text, suitable for a README:


## Table of Contents

1. [Purpose](#1-purpose)
2. [Features](#2-features)
    - [Question Answering (QA) Across CVs](#21-question-answering-qa-across-cvs)
    - [CV Summarization](#22-cv-summarization)
    - [Candidate Comparison and Skill Analysis](#23-candidate-comparison-and-skill-analysis)
    - [CV Ingestion and File Handling](#24-cv-ingestion-and-file-handling)
    - [LLM Integration (Local)](#25-llm-integration-local)
    - [Web Interface](#26-web-interface)
3. [Installation and Setup](#3-installation-and-setup)
    - [Requirements](#31-requirements)
    - [Step-by-Step Guide](#32-step-by-step-guide)
4. [Project Structure](#4-project-structure)
5. [Problem Solving and Challenges](#5-problem-solving-and-challenges)
    - [Challenge: Chunking CVs for RAG](#51-challenge-chunking-cvs-for-rag)
    - [Challenge: Making TF-IDF Skill Scoring Work](#52-challenge-making-tf-idf-skill-scoring-work)
6. [Final Thoughts](#6-final-thoughts)

<hr style="height:2px; background-color:#ccc; border:none;" />

## 1. Purpose

Hiring is time-consuming. Recruiters often spend hours reading and comparing CVs, trying to extract relevant insights under pressure.

**Smart Recruiter Assistant** streamlines this process by offering a **natural, interactive interface** powered by LLMs. It lets recruiters:

* Upload multiple candidate CVs
* Ask questions in plain language
* Receive detailed, context-aware answers
* Compare candidates based on specific criteria
* Extract insights and summaries instantly

This tool isn’t just a static demo — it works on live, user-uploaded data. While not fully production-ready yet, it's a **functional prototype** designed for real-world use and feedback.

<hr style="height:2px; background-color:#ccc; border:none;" />

## 2. Features

The **Smart Recruiter Assistant** is a robust RAG-based application designed to streamline the candidate evaluation process. It combines traditional NLP techniques with the reasoning power of LLMs to help recruiters make informed decisions faster. Here’s a breakdown of what it offers:

### 2.1 Question Answering (QA) Across CVs

* Allows recruiters to ask natural language questions (e.g., *“Who has experience with deploying machine learning models?”*).
* Uses semantic similarity and RAG techniques to retrieve relevant chunks from all CVs and respond accurately.
* Answers are factual, pulled directly from content — no hallucinated information.

### 2.2 CV Summarization

* Automatically generates concise summaries of each candidate's CV using an LLM.
* Focuses on factual information, highlighting key qualifications, experience, and skill sets.
* Ignores filler content (like over-hyped objectives) to provide clean, useful summaries.

### 2.3 Candidate Comparison and Skill Analysis

* While not comparing candidates side-by-side directly, the assistant allows for:

  * **Skill-specific questioning**, like *“Who has more experience in deep learning?”*
  * **Term-frequency-based skill scoring**, ranking candidates by how prominently a skill is mentioned.
* Great for identifying strengths and weaknesses in technical areas at a glance.
* Interactive visualization provides clear and insightful visualizations of skill relevance using **Plotly** and **Gradio**.
* Recruiters can:

  * Select a skill and see which candidates emphasize it most.
  * Select a candidate and explore which skills are most (or least) mentioned in their CV.

### 2.4 CV Ingestion and File Handling

* Accepts multiple formats: `.txt`, `.pdf`, `.docx`, etc.
* CVs can be uploaded at any time during a session — doesn’t require batch processing.
* All uploaded CVs are automatically parsed and indexed for analysis.

### 2.5 LLM Integration (Local)

* Powered by **LLaMA 3.1**, running via **Ollama** for fast local inference.
* No API tokens or cloud costs — great for development and experimentation.
* Not yet optimized for production, but it runs on real, live recruiter data.

### 2.6 Web Interface

* Built with **Gradio**, the UI is clean, tabbed, and responsive.
* Different tabs serve different goals: summarization, skill assessment, and analysis.
* Users interact naturally — no coding or manual prompt-writing required.

<hr style="height:2px; background-color:#ccc; border:none;" />

## 3. Installation and Setup

Follow these steps to get the **Smart Recruiter Assistant** running locally.

### 3.1 Requirements

* **Python** 3.10+
* **pip** (comes with Python)
* [**Ollama**](https://ollama.com/) (for running LLaMA 3.1 locally)
* Recommended: Use a **virtual environment** (`venv` or `conda`) to isolate dependencies

---

### 3.2 Step-by-Step Guide

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/smart-recruiter-assistant.git
cd smart-recruiter-assistant
```

#### 2. Create a Virtual Environment (Recommended)

```bash
# With venv
py -3.10 -m venv app-env
app-env\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Install and Start Ollama

* Follow official setup: [https://ollama.com/download](https://ollama.com/download)
* Pull and run LLaMA 3.1 model:

```bash
ollama pull llama3.1
ollama run llama3.1
```

Ensure Ollama is running in the background.

#### 5. Run the App

```bash
python run.py
```

* The app will start on a **local Gradio interface**
* Necessary directories will be created automatically (e.g., for uploads and logs)
* No manual `.env` editing is needed — a default file is already included


<hr style="height:2px; background-color:#ccc; border:none;" />

## 4. Project Structure

Below is a breakdown of the project’s folder layout and what each component does:

```
├── app/                     # Core application logic
│   ├── main.py              # Entry point for launching the Gradio interface
│   ├── chatbot.py           # Handles question-answering and RAG logic
│   ├── summarizer.py        # Summarization and insight extraction using the LLM
│   ├── skill_assessor.py    # Skill scoring logic using classic NLP (TF/TF-IDF)
│   └── utils/               # Utility functions and shared components
│       ├── config.py        # Configuration and environment variables
│       ├── embedding.py     # Loads vector store and handles embeddings
│       ├── parser.py        # Preprocessing, normalization, tokenization
│       ├── logger.py        # Logging utilities
│       ├── llm_extractor.py # Low-level wrappers around LLM usage
│       └── callbacks.py     # Custom callbacks (for streaming, etc.)

├── data/                    # Persistent data storage
│   ├── txt_cvs/             # Parsed candidate CVs (.txt format)
│   ├── uploads/             # Raw uploaded files from users
│   ├── vector_db/           # Chroma-based vector index and metadata
│   └── logs/                # Logs of user queries and app events

├── .env                     # Environment config (safe to leave as-is)
├── .gitignore               # Git ignore file
├── requirements.txt         # All necessary dependencies
├── README.md                # You're reading it
└── run.py                   # Launch script to start the app
```

This modular structure ensures separation of concerns:

* LLM logic is decoupled from interface logic.
* Preprocessing and vector indexing are easily swappable.
* You can plug in new models or UI layers with minimal rewiring.

<hr style="height:2px; background-color:#ccc; border:none;" />

## 5. Problem Solving and Challenges

Building a system like the Smart Recruiter Assistant wasn't just a matter of connecting APIs or following a standard Retrieval-Augmented Generation (RAG) recipe. Many of the real challenges emerged from edge cases, subtle model behaviors, and the friction between theory and practical results. Below are some of the major hurdles encountered during development, how I approached them, and what lessons were learned.

---

### 5.1 Challenge: Chunking CVs for RAG

One of the most deceptively tricky parts of building this project was **chunking candidate CVs** for retrieval. It sounds straightforward — split a document, embed chunks, search with a query — but it turned into a war of trade-offs, hallucinations, and weird edge cases.

#### Attempt 1: Standard Chunking with a Catch

I started off with **traditional chunking**: breaking each CV into overlapping text chunks using a fixed size and stride. This works well on many documents, and retrieval quality seemed decent at first. But something felt off during testing. When I asked the assistant about one candidate, it would sometimes mention achievements or skills that clearly belonged to another.

After investigating, I found the root cause: **information bleeding**. Since the embedding store wasn’t isolating content per CV, and the chunks didn’t contain clear ownership metadata, **similar content (like shared project names or job titles)** would leak across candidates.

This wasn’t a bug — it was a design flaw. And it led me to rethink how context is grouped.

#### Attempt 2: Structured LLM Rewriting + Sectional Chunking

To fix this, I tried a different idea. Instead of raw chunking, I **preprocessed each CV through an LLM**. I prompted it to:

* **Extract and restructure** the content into a unified format (clearly labeled sections: Skills, Education, Experience, Projects).
* **Repeat the candidate's name** in each section to preserve ownership in downstream context.

The output was saved as `.txt`, then chunked by section headers rather than blind tokens. This gave me cleaner retrieval units with **section-specific semantics** and reduced ambiguity about which candidate the chunk belonged to.

It looked brilliant — until it wasn’t.

#### The Unexpected Consequences

* **Longer sections like "Experience" were under-ranked** by the retriever because embedding similarity punishes larger, noisier text blocks.
* **Smaller, often irrelevant chunks** were being returned instead, just because they had higher similarity scores (cosine similarity loves short, dense vectors).
* And most painfully: **LLM conversion was slow** and sometimes lossy. Important information occasionally got reworded or omitted.

At this point, I had:

* A pretty formatter that wasn't scalable.
* A retriever that valued brevity over depth.
* A growing frustration that LLMs were adding more noise than signal.

#### Attempt 3: Realistic RAG with Metadata and Prompt Engineering

Time to go back — but smarter.

I returned to **regular chunking**, but this time, I added two critical upgrades:

1. **Metadata tagging**: Each chunk was saved with a `candidate_id` key, preserving document identity.
2. **Prompt-level grouping**: During retrieval, instead of sending raw chunks, I **grouped all retrieved chunks by candidate**, and injected that grouping into the prompt:

   > "Below are excerpts from the CV of \[Candidate Name]. Answer the recruiter’s question based on this candidate's information only..."

This struck the perfect balance:

* It preserved chunk size consistency (for better embedding quality).
* It avoided cross-contamination.
* It was fast and robust, with **no need for costly LLM restructuring**.

#### Bonus Insight: Getting the Candidate Name

To attach the correct metadata, I needed each candidate’s name. At first, I thought about extracting it from filenames — but filenames were messy and inconsistent. So, I sent each uploaded CV to the LLM with a simple prompt:

> "Extract the full name of the candidate from this resume."

Once identified, the CV was renamed and saved as a `.txt` with the correct name. This allowed me to **reliably tag chunks** and use the name throughout the prompt.

#### Debugging and Logging

To make sure everything worked smoothly, I added:

* Logging during chunk creation (what chunk came from which file).
* Logging during retrieval (which chunks were fetched, and from which candidate).
  This gave me full visibility into what the assistant was actually seeing and using to answer questions.


Perfect — here’s the next subsection for the **Problem Solving and Challenges** section, focused on your TF-IDF challenges. It keeps the tone consistent with the previous one and walks through the issue like a story.

---

### 5.2 Challenge: Making TF-IDF Skill Scoring Work

Using TF-IDF might sound like a plug-and-play solution for keyword relevance — but in practice, it was anything but.

#### Problem 1: Special Character Skills Not Detected

Right out of the gate, I noticed something weird. Terms like `C++` were returning **zero scores**, even though they clearly appeared in the CVs. The reason? The default tokenizer behind `TfidfVectorizer` silently dropped symbols like `+`, `#`, or even dots. It was never trained to handle skill names — only “words.”

I realized quickly that I needed to take control of the **text preprocessing and tokenization**.

#### Fixing the Preprocessing

To handle this:

* I wrote a **custom preprocessor** using regex and string operations.
* Lowercased all text.
* Preserved key characters like `+`, `#`, and `.` — because these matter in tech (`C++`, `.NET`, `Node.js`).
* Replaced dashes (`-`) and underscores (`_`) with spaces to unify terms like:

  * `"object-oriented programming"` → `"object oriented programming"`
  * `"machine_learning"` → `"machine learning"`
* Cleaned up trailing punctuation by **removing periods only when they appear at the end of a word** (so `.NET` is preserved, but `"Python."` becomes `"Python"`).

These refinements made token matching much more accurate, and skills like `C++` finally showed up as valid TF-IDF terms.

#### Problem 2: Misleading Score Interpretation

Once tokenization was fixed, another issue surfaced: **the numbers weren’t helpful**. Raw TF-IDF scores varied wildly and didn’t mean much to a recruiter. A term could appear once and get a higher score than another term that appeared ten times in another CV — due to inverse document frequency.

I tried:

* **Min-max normalization** — but it exaggerated differences. A single over-mention of a term would make that CV look way better than others.
* **Custom scalers** that preserved zeros but squashed everything else — still confusing.

Eventually, I took a step back and asked: *what are we trying to show here?*

#### The Decision: Go Raw and Be Honest

Instead of bending over backwards to normalize, I went for **raw term frequency** — just counting how many times the word appears.

It’s not fancy, but it’s **honest**:

* 0 → term not mentioned
* 1+ → term appears N times

This made the number **directly interpretable** and easier to explain. I added a clear disclaimer: *this number only represents how many times the skill is mentioned — not whether the candidate actually possesses the skill.*

#### Bonus: Interactivity & Visualization

To make the data usable, I plugged these scores into an **interactive visualization**:

* One tab showed **skill-by-skill**, allowing recruiters to drill into how candidates stack up for a given skill.
* Another tab flipped the view, showing a single candidate and all their skill scores.

Combined with a good interface and some dark-mode aesthetics, it became a useful feature — even if it didn’t try to predict competence. It helped recruiters **filter noise and spot possible fits** faster.

<hr style="height:2px; background-color:#ccc; border:none;" />

## 6. Final Thoughts

The **Smart Recruiter Assistant** isn’t just a proof of concept — it’s a serious step toward modernizing how hiring decisions are made. In an age where recruiters are bombarded with countless resumes, this tool offers a way to **cut through the noise**, highlight what's relevant, and **make the hiring process more efficient and informed**.

By combining classical NLP techniques like TF-IDF with cutting-edge retrieval-augmented generation (RAG), and deploying it all through an interactive web interface powered by LLaMA 3.1, this assistant aims to bridge the gap between **raw information** and **practical insight**.

Of course, no tool replaces human judgment — and that’s the point. This assistant is **not a decision-maker**, but a **decision-support system**. It helps recruiters ask better questions, find better matches, and spend more time on what really matters: the people behind the papers.

This project was built with care, curiosity, and a lot of trial and error. The result is a tool that works on real-world data, with real benefits, and room to grow into a production-grade solution.

Whether you’re a developer looking to improve it, or a recruiter curious to test it, **welcome aboard** — and thank you for reading this far.
