import os
from langchain_classic.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
from langchain_community.retrievers import CohereRagRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.embeddings import CohereEmbeddings
from langchain_chroma import Chroma
import time

from dotenv import load_dotenv
load_dotenv()

# Load API
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]

def delivery_method_predictor_1(profile: dict):

    # ----------------------------------
    # 1. Load Vector DB
    # ----------------------------------
    cohere_embeddings = CohereEmbeddings(user_agent="hhh")
    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=cohere_embeddings
    )

    # ----------------------------------
    # 2. Structured Business Profile Query
    # ----------------------------------
    structured_profile = f"""
    BUSINESS PROFILE:

    Industry: {profile.get("industry")}
    Primary Goal: {profile.get("goal")}
    Desired AI Outcome: {profile.get("use_case")}
    AI Experience Level: {profile.get("experience")}
    Available Data Size: {profile.get("data_size")}
    Frontend Deployment Preference: {profile.get("front_deployment")}
    Backend Deployment Preference: {profile.get("back_deployment")}
    """

    # ----------------------------------
    # 3. System Prompt
    # ----------------------------------
    system_prompt = """
    You are an AI Solution Architect.

    You MUST strictly use ONLY the retrieved knowledge base entries below.
    If information is not present in the retrieved context, you must not assume it.

    The database entries contain these fields:
    - Delivery Method
    - Simple Description
    - Typical Best For
    - Hosting Control
    - Complexity

    INSTRUCTIONS:

    1. Compare the BUSINESS PROFILE with each retrieved Delivery Method.
    2. Evaluate alignment based on:
    - Industry & Goal fit (Typical Best For)
    - Data size vs Complexity
    - Deployment vs Hosting Control
    - AI Experience vs Complexity
    3. Select ONE best Delivery Method.
    4. Provide reasoning using ONLY retrieved fields.
    5. If no strong alignment exists, say so clearly.

    Return output in this structured format:

    Recommended Delivery Method:
    Why This Fits:
    - Industry & Goal Alignment:
    - Data & Complexity Alignment:
    - Hosting Alignment:
    Alternative Option:

    Retrieved Knowledge Base:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # ----------------------------------
    # 4. LLM (Lower Temperature for Stability)
    # ----------------------------------
    llm = ChatCohere(temperature=0.1)

    # ----------------------------------
    # 5. Better Retrieval (k=2)
    # ----------------------------------
    retriever = db.as_retriever(search_kwargs={"k": 2})

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    result = rag_chain.invoke({"input": structured_profile})

    return result["answer"]




def delivery_method_predictor_2(profile: dict):

    # ----------------------------------
    # 1. Format Business Profile
    # ----------------------------------
    structured_profile = f"""
BUSINESS PROFILE:

Industry: {profile.get("industry")}
Primary Goal: {profile.get("goal")}
Desired AI Outcome: {profile.get("use_case")}
AI Experience Level: {profile.get("experience")}
Available Data Size: {profile.get("data_size")}
Frontend Deployment Preference: {profile.get("front_deployment")}
Backend Deployment Preference: {profile.get("back_deployment")}
"""

    # ----------------------------------
    # 2. All 12 Delivery Options Embedded Directly
    # ----------------------------------
    delivery_options = """
AVAILABLE DELIVERY METHODS:

#1
Delivery Method: AI SaaS / Out-of-the-Box
Simple Description: Plug-and-play AI software with no build required
Typical Best For: Fast business improvements, automation, content
Hosting Control: Vendor
Complexity: Low

#2
Delivery Method: Proprietary Model Provider (API)
Simple Description: Direct use of closed AI models via API
Typical Best For: Custom features without infrastructure
Hosting Control: Vendor
Complexity: Low-Medium

#3
Delivery Method: Proprietary Model + RAG
Simple Description: API model connected to your documents/data
Typical Best For: Knowledge search, support bots, internal Q&A
Hosting Control: Vendor model + your data
Complexity: Medium

#4
Delivery Method: Fine-Tuned Proprietary Model
Simple Description: Provider model trained on your dataset
Typical Best For: Brand voice, classification, specialized outputs
Hosting Control: Vendor
Complexity: Medium

#5
Delivery Method: Open-Source Model (Prompt-Only)
Simple Description: Self-hosted open model with prompting only
Typical Best For: Privacy-sensitive or low-cost internal AI
Hosting Control: You
Complexity: Medium

#6
Delivery Method: Fine-Tuned Open-Source Model
Simple Description: Open model trained on your proprietary data
Typical Best For: Higher accuracy with full control
Hosting Control: You
Complexity: Medium-High

#7
Delivery Method: Open-Source Model + RAG
Simple Description: Self-hosted model connected to internal knowledge
Typical Best For: Private document search, secure copilots
Hosting Control: You
Complexity: High

#8
Delivery Method: Fine-Tuned / Custom Model + RAG
Simple Description: Tuned model + retrieval for maximum accuracy
Typical Best For: Enterprise knowledge systems, regulated AI
Hosting Control: You
Complexity: High

#9
Delivery Method: From-Scratch Custom Model
Simple Description: Fully new model trained from the ground up
Typical Best For: Research, unique IP, extreme specialization
Hosting Control: You
Complexity: Very High

#10
Delivery Method: Hybrid / Ensemble Architecture
Simple Description: Multiple AI models working together
Typical Best For: Complex, multi-step, or multimodal workflows
Hosting Control: Mixed
Complexity: High

#11
Delivery Method: Embedded / Edge AI
Simple Description: AI running on local devices or hardware
Typical Best For: Real-time, offline, IoT, manufacturing
Hosting Control: Local device
Complexity: High

#12
Delivery Method: Multi-Layer AI System
Simple Description: Stacked combination of SaaS, RAG, APIs, edge, etc.
Typical Best For: Large-scale enterprise AI ecosystems
Hosting Control: Mixed
Complexity: Very High
"""

    # ----------------------------------
    # 3. Strong Comparison Prompt
    # ----------------------------------
    system_prompt = """
You are an AI Solution Architect.

You must compare the BUSINESS PROFILE with ALL 12 available delivery methods.

INSTRUCTIONS:

1. Evaluate each option against:
   - Industry fit
   - Goal alignment
   - Data scale alignment
   - AI experience vs Complexity
   - Deployment vs Hosting Control
2. Select ONE best overall option.
3. Provide concise reasoning.
4. Do not invent new delivery methods.
5. Only choose from the provided 12 options.

Return output in this format:

Recommended Delivery Method:
Why This Fits:
- Industry & Goal Alignment:
- Data & Complexity Alignment:
- Hosting Alignment:
Alternative Option:
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    llm = ChatCohere(temperature=0.1)

    chain = prompt | llm

    result = chain.invoke({
        "input": structured_profile + "\n\n" + delivery_options
    })

    return result.content

