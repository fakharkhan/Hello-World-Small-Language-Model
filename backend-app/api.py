from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from main import SmallLanguageModel
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
tokenizer = None
model = None
qa_pairs = []

def load_qa_pairs():
    global qa_pairs
    try:
        with open('dataset.txt', 'r', encoding='utf-8') as f:
            current_question = None
            current_answer = None
            
            for line in f:
                line = line.strip()
                if line.startswith('Question:'):
                    if current_question and current_answer:
                        qa_pairs.append((current_question, current_answer))
                    current_question = line[len('Question:'):].strip()
                    current_answer = None
                elif line.startswith('Answer:'):
                    current_answer = line[len('Answer:'):].strip()
            
            if current_question and current_answer:
                qa_pairs.append((current_question, current_answer))
        
        logger.info(f"Loaded {len(qa_pairs)} QA pairs from dataset")
    except Exception as e:
        logger.error(f"Error loading QA pairs: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    global tokenizer, model
    try:
        logger.info("Loading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer configured successfully")
        
        logger.info("Loading model...")
        model = SmallLanguageModel()
        model.load_state_dict(torch.load('qa_model.pth', map_location=torch.device('cpu')))
        model.eval()
        logger.info("Model loaded successfully")
        
        logger.info("Loading QA pairs...")
        load_qa_pairs()
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

class Question(BaseModel):
    text: str

def find_best_match(question):
    global qa_pairs
    
    # First try exact match
    for q, a in qa_pairs:
        if question.lower() in q.lower():
            return a
    
    # Try word matching
    question_words = set(question.lower().split())
    best_match = None
    max_overlap = 0
    
    for q, a in qa_pairs:
        q_words = set(q.lower().split())
        overlap = len(question_words.intersection(q_words))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = a
    
    return best_match if max_overlap > 1 else None

def generate_answer(model, tokenizer, question, max_length=100):
    # First try to find a match in the dataset
    dataset_answer = find_best_match(question)
    if dataset_answer:
        return dataset_answer
    
    # If no match found, generate an answer
    input_text = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    
    generated_tokens = input_ids
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated_tokens, attention_mask)
            next_token_logits = outputs[:, -1, :]
            
            # Apply temperature sampling
            probs = F.softmax(next_token_logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1,1))], dim=1)
    
    answer = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    try:
        answer = answer.split("Answer:", 1)[1].strip()
    except IndexError:
        answer = answer.strip()
    
    return answer

@app.post("/api/ask")
async def ask_question(question: Question):
    global tokenizer, model
    
    if not tokenizer or not model:
        logger.error("Model or tokenizer not initialized")
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        logger.info(f"Received question: {question.text}")
        
        # Generate answer
        logger.info("Generating answer...")
        answer = generate_answer(model, tokenizer, question.text)
        
        # Post-process the answer
        if not answer or len(answer.strip()) < 10:
            answer = "I apologize, but I don't have enough information to provide a meaningful answer to that question."
        
        logger.info(f"Generated answer: {answer}")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    global tokenizer, model, qa_pairs
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "qa_pairs_loaded": len(qa_pairs)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 