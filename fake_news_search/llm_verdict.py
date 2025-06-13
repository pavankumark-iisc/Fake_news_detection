from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnableSequence

# Load sentence transformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load Qwen model
MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto"
)

# Optional: disable cache if needed (not required for Qwen, but safe to include)
model.generation_config = GenerationConfig.from_pretrained(
    MODEL_NAME, trust_remote_code=True
)
model.generation_config.use_cache = False

# Define pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=150
)

# Wrap in LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["claim", "verdict", "score"],
    template="""
You are a helpful assistant that explains fake news detection results to the user.

Claim: "{claim}"
Detection Result: "{verdict}"
Similarity Score: {score}

Explain the verdict in simple and polite terms suitable for an end user. Avoid technical jargon.
""",
)
chain = prompt_template | llm

# Function to compute verdict and generate explanation
def compute_verdict_and_explanation(claim: str, articles: list) -> dict:
    claim_emb = embedder.encode(claim, convert_to_tensor=True)

    scores = []
    for article in articles:
        raw_text = article.get("snippet", "").strip()
        if raw_text:
            doc_emb = embedder.encode(raw_text, convert_to_tensor=True)
            score = util.cos_sim(claim_emb, doc_emb).item()
            scores.append(score)
        else:
            scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    verdict = "Likely Real" if avg_score > 0.6 else "Potentially Fake"

    explanation = chain.invoke({
    "claim": claim,
    "verdict": verdict,
    "score": round(avg_score, 2)
    })

    if isinstance(explanation, dict):
      explanation_text = explanation.get("text", "").strip()
    else:
      explanation_text = explanation.strip()



    return {
        "verdict": verdict,
        "score": round(avg_score, 2),
        "explanation": explanation_text

    }
