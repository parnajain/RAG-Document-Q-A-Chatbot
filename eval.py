import pandas as pd
import time
from langchain_core.prompts import PromptTemplate
from rag_pipeline import (
    get_embeddings,
    get_llm,
    load_documents,
    build_vector_store,
    build_retriever,
    build_chain
)

# Prompt Template
prompt_template = """
You are a document assistant.

RULES:
1. Answer only using the provided context.
2. If the user asks for a summary, create a summary from the context.
3. If the answer cannot be found in the context, respond:

Out of scope: The answer is not present in the uploaded document.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

results=[]
benchmark = pd.read_csv("tests.csv")
pdf_paths = [
    "Disk management.pdf"
]

chunks = load_documents(pdf_paths)


def evaluate_answer(llm, query, ground_truth, answer, source_docs):

    context = "\n\n".join([doc.page_content for doc in source_docs])

    faithfulness_prompt = f"""
You are an evaluator.

Context:
{context[:3000]}

Answer:
{answer}

Is every factual claim in the answer supported by the context?

Return ONLY a decimal between 0.0 and 1.0.

0.0 = hallucinated
0.5 = partially supported
1.0 = fully supported

Do not explain.
"""

    relevancy_prompt = f"""
Question:
{query}

Answer:
{answer}

Rate how well the answer addresses the question.

0.0 = completely irrelevant
0.25 = mostly irrelevant
0.5 = partially relevant
0.75 = mostly relevant
1.0 = completely relevant

Reply ONLY with one number:
0.0
0.25
0.5
0.75
or
1.0

Do not explain.
"""

    context_prompt = f"""
Question:
{query}

Retrieved Context:
{context[:3000]}

Can this context alone answer the question?

Score:

0.0 = not useful
0.25 = slightly useful
0.5 = partially useful
0.75 = mostly sufficient
1.0 = fully sufficient

Reply ONLY with one number:
0.0
0.25
0.5
0.75
or
1.0

Do not explain.
"""

    accuracy_prompt = f"""
You are evaluating a Question Answering system.

Question:
{query}

Ground Truth:
{ground_truth}

Generated Answer:
{answer}

Evaluate how semantically similar the generated answer is to the ground truth.

Scoring Guidelines:

1.0 = Meaning is completely equivalent or mostly correct. All important information is present. 
Minor details are missing or wording differs.

0.6 = Partially correct. Some important information is missing.

0.4 = Only a small part of the answer is correct.

0.2 = Mostly incorrect.

0.0 = Completely incorrect or unrelated.

Return ONLY one decimal number between 0.0 and 1.0.

Do not explain your score.
"""

    def get_score(prompt):
        try:
            return float(llm.invoke(prompt).content.strip())
        except:
            return 0.0

    return {
        "Faithfulness": get_score(faithfulness_prompt),
        "Answer Relevancy": get_score(relevancy_prompt),
        "Context Quality": get_score(context_prompt),
        "Accuracy": get_score(accuracy_prompt)
    }

embedding_models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5"
]

strategies = [
    "Chroma MMR",
    "Ensemble BM25 + Chroma"
]

llm = get_llm()
llm.invoke("Hello")
for embedding_model in embedding_models:

    embedding = get_embeddings(embedding_model)
    vector_store = build_vector_store(embedding, chunks, embedding_model)

    for retrieval_strategy in strategies:

        retriever = build_retriever(vector_store, chunks, retrieval_strategy)
        qa_chain = build_chain(llm, retriever, PROMPT)

        for _, row in benchmark.iterrows():

            question = row["question"]
            ground_truth = row["ground_truth"]

            # Measure latency
            times = []
            for _ in range(3):
                start = time.time()
                result = qa_chain.invoke({"query": question})
                times.append(time.time() - start)

            latency = sum(times) / len(times)

            answer = result["result"]
            source_docs = result["source_documents"]

            scores = evaluate_answer(
                llm,
                question,
                ground_truth,
                answer,
                source_docs
            )
            results.append({
                "Embedding": embedding_model,
                "Retriever": retrieval_strategy,
                "Question": question,
                "Ground Truth": ground_truth,
                "Answer": answer,
                "Accuracy": scores["Accuracy"],
                "Faithfulness": scores["Faithfulness"],
                "Answer Relevancy": scores["Answer Relevancy"],
                "Context Quality": scores["Context Quality"],
                "Latency": latency
            })

df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)
summary = (
    df.groupby(["Embedding", "Retriever"])
      .agg(
          Correct=("Accuracy", "sum"),
          Total=("Accuracy", "count"),
          Accuracy=("Accuracy", "mean"),
          Faithfulness=("Faithfulness", "mean"),
          Answer_Relevancy=("Answer Relevancy", "mean"),
          Context_Quality=("Context Quality", "mean"),
          Latency=("Latency", "mean")
      )
      .round(3)
      .reset_index()
)

print(summary)

summary.to_csv("summary_results.csv", index=False)