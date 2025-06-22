from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.prompts import PromptTemplate  # Add this import
import numpy as np  # Add this import if not present
import uvicorn
import warnings
import os
from difflib import get_close_matches
import config

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize FastAPI app
app = FastAPI(
    title="MRPH Chatbot API",
    description="API for the MRPH customer service chatbot",
    version="1.0.0"
)

# Initialize OpenAI client
llm = OpenAI(api_key=config.OPENAI_API_KEY, temperature=0, max_tokens=30)

# Initialize MongoDB and vector store
client = MongoClient(config.MONGODB_CONN_STRING)
collection = client[config.DB_NAME][config.COLLECTION_NAME]
embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
vector_store = MongoDBAtlasVectorSearch(
    collection, 
    embeddings, 
    index_name=config.INDEX_NAME,
)

# Basic conversation patterns
BASIC_CONVERSATIONS = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there! How may I assist you?",
    "hey": "Hey! What can I do for you?",
    "bye": "Goodbye! Have a great day!",
    "thanks": "You're welcome! Let me know if you need anything else.",
    "thank you": "You're welcome! Is there anything else I can help you with?",
    "how are you": "I'm doing well, thank you! How can I assist you today?",
    "good morning": "Good morning! How may I help you?",
    "good afternoon": "Good afternoon! What can I do for you?",
    "good evening": "Good evening! How can I assist you?"
}

MRPH_CATEGORIES = {
    'general': [
        'mrph', 'app', 'about', 'what', 'purpose', 'does', 'marketplace', 
        'who', 'use', 'age', 'help', 'support', 'assistance'
    ],
    'registration': [
        'register', 'signup', 'sign up', 'create', 'account', 'customer', 
        'engineer', 'professional', 'trade certification', 'information', 
        'required', 'registration', 'details', 'role', 'selection', 'choose'
    ],
    'authentication': [
        'log','login', 'log in', 'sign in', 'access', 'password', 'forgot', 
        'reset', 'recover', 'logout', 'log out', 'sign out', 'exit', 
        'change', 'security', 'verify', 'verification', 'email', 'phone'
    ],
    'services': [
        'electrical', 'hvac', 'plumbing', 'security', 'power', 'services', 
        'available', 'categories', 'repair', 'maintenance', 'installation',
        'leak', 'covers', 'systems', 'solutions'
    ],
    'booking': [
        'book', 'booking', 'request', 'quote', 'price', 'estimate', 'service', 
        'hire', 'schedule', 'appointment', 'cancel', 'cancellation', 'reschedule', 
        'time', 'date', 'confirmation', 'confirm'
    ],
    'engineer_workflow': [
        'jobs', 'available', 'requests', 'accept', 'take', 'work', 'respond', 
        'bid', 'offer', 'view', 'details', 'send', 'availability', 'schedule', 
        'set', 'completed', 'projects', 'portfolio'
    ],
    'payment': [
        'pay', 'payment', 'stripe', 'billing', 'methods', 'card', 'extra', 
        'additional', 'price', 'cost', 'money', 'paid', 'refund', 'money back', 
        'return', 'charge', 'invoice'
    ],
    'subscription': [
        'subscription', 'plans', 'upgrade', 'pro', 'premium', 'features', 
        'free', 'plan', 'manage', 'billing'
    ],
    'service_management': [
        'status', 'statuses', 'pending', 'progress', 'complete', 'track', 
        'after', 'next', 'steps', 'remove', 'history', 'past', 'previous', 
        'my', 'finished'
    ],
    'communication': [
        'message', 'chat', 'communicate', 'contact', 'messages', 'conversation', 
        'talk', 'speak', 'discuss', 'assistant', 'chatbot'
    ],
    'reviews': [
        'review', 'rating', 'feedback', 'reviews', 'ratings', 'rate', 'stars', 
        'excellent', 'good', 'average', 'poor', 'leave', 'write'
    ],
    'notifications': [
        'notifications', 'alerts', 'updates', 'receive', 'check', 'see', 
        'not receiving', 'missing', 'notify', 'alert'
    ],
    'profile': [
        'profile', 'update', 'edit', 'change', 'settings', 'information', 
        'details', 'my list', 'favourites', 'saved', 'bookmarks', 'personal'
    ],
    'policies': [
        'terms', 'policy', 'privacy', 'data', 'rules', 'legal', 'policies', 
        'refund', 'conditions', 'agreement'
    ],
    'navigation': [
        'navigation', 'tabs', 'menu', 'home', 'screen', 'dashboard', 'welcome', 
        'search', 'find', 'browse', 'access', 'features', 'bottom'
    ],
    'troubleshooting': [
        'problem', 'issue', 'bug', 'error', 'not working', 'technical', 
        'issues', 'problems', 'fix', 'solve', 'trouble'
    ],
    'getting_started': [
        'new', 'first time', 'getting started', 'begin', 'start', 'tutorial', 
        'guide', 'how', 'setup'
    ]
}

class Query(BaseModel):
    question: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    return {"message": "Welcome to MRPH Chatbot API"}

def calculate_similarity_score(question: str, faq_question: str) -> float:
    """Calculate similarity score between user question and FAQ question"""
    try:
        # Get embeddings for both questions
        q1_embedding = embeddings.embed_query(question)
        q2_embedding = embeddings.embed_query(faq_question)
        
        # Calculate cosine similarity
        similarity = np.dot(q1_embedding, q2_embedding) / (
            np.linalg.norm(q1_embedding) * np.linalg.norm(q2_embedding)
        )
        return float(similarity)
    except Exception:
        return 0.0

def is_similar_to_keywords(word: str, keywords: list, threshold: float = 0.8) -> bool:
    """Check if word is similar to any keyword using fuzzy matching"""
    for keyword in keywords:
        matches = get_close_matches(word, [keyword], n=1, cutoff=threshold)
        if matches:
            return True
    return False

def is_topic_relevant(question: str, threshold: float = 0.8) -> tuple[bool, str]:
    """
    Check if question is relevant to MRPH and return category
    Returns: (is_relevant, category)
    """
    question_lower = question.lower()
    for category, keywords in MRPH_CATEGORIES.items():
        for keyword in keywords:
            if keyword in question_lower:
                return True, category
    return False, ""

@app.post("/chat", response_model=ChatResponse)
async def chat(query: Query):
    try:
        question = query.question.strip().lower()

        # Restrict question to 100 words before embedding
        if len(question.split()) > 100:
            return {"answer": "Your question is too long. Please limit your query to 100 words or less."}

        # Handle basic conversations
        if question in BASIC_CONVERSATIONS:
            return {"answer": BASIC_CONVERSATIONS[question]}
            
        # Check topic relevance
        is_relevant, category = is_topic_relevant(question)
        if not is_relevant:
            return {
                "answer": "I can only help with MRPH related questions about:"
                         "\n- Our services (electrical, HVAC, plumbing, etc.)"
                         "\n- Account & registration"
                         "\n- Booking appointments"
                         "\n- Payments & billing"
                         "\n- Technical support"
            }

        # Check if question is related to MRPH topics
        MRPH_keywords =  [
    'mrph', 'about', 'app', 'purpose', 'marketplace', 'log',
    'engineers', 'services', 'who', 'use', 'age', 'customers', 'create', 
    'account', 'register', 'sign up', 'customer', 'engineer', 'professional', 
    'trade certification', 'information', 'required', 'registration', 'details', 
    'role', 'selection', 'choose', 'log in', 'login', 'sign in', 'access', 
    'forgot', 'password', 'reset', 'recover', 'log out', 'logout', 'sign out', 
    'exit', 'change', 'profile', 'update', 'security', 'available', 'electrical', 
    'hvac', 'plumbing', 'categories', 'request', 'quote', 'price', 'estimate', 
    'service', 'book', 'hire', 'select', 'offer', 'view', 'details', 'my', 
    'status', 'jobs', 'requests', 'accept', 'take', 'work', 'respond', 'bid', 
    'extra', 'payment', 'additional', 'pay', 'stripe', 'billing', 'methods', 
    'card', 'payments', 'add', 'subscription', 'plans', 'upgrade', 'pro', 
    'premium', 'features', 'statuses', 'pending', 'progress', 'complete', 
    'track', 'after', 'booking', 'next', 'steps', 'reschedule', 'time', 
    'date', 'cancel', 'cancellation', 'remove', 'history', 'past', 'previous', 
    'message', 'chat', 'communicate', 'contact', 'assistant', 'chatbot', 
    'help', 'leave', 'review', 'rating', 'feedback', 'reviews', 'ratings', 
    'shows', 'display', 'rate', 'notifications', 'receive', 'alerts', 'updates', 
    'check', 'see', 'not receiving', 'missing', 'contains', 'settings', 'edit', 
    'manage', 'my list', 'favourites', 'saved', 'bookmarks', 'terms', 'rules', 
    'policy', 'privacy', 'data', 'refund', 'money back', 'return', 'navigation', 
    'tabs', 'menu', 'bottom', 'home', 'screen', 'dashboard', 'welcome', 'search', 
    'find', 'browse', 'verification', 'verify', 'email', 'phone', 'secure', 
    'safety', 'safe', 'support', 'assistance', 'not working', 'problem', 'issue', 
    'bug', 'technical', 'issues', 'problems', 'new', 'first time', 'getting started', 
    'begin', 'then', 'availability', 'schedule', 'set', 'completed', 'projects', 
    'past work', 'portfolio', 'paid', 'money', 'confirmation', 'process','confirm'
]
        
        # Split question into words and check each word for similarity
        question_words = question.split()
        is_relevant = any(
            is_similar_to_keywords(word, MRPH_keywords)
            for word in question_words
        )
        
        if not is_relevant:
            return {"answer": "I can only help with MRPH related questions. Please ask about our services, booking, registration, or technical support or try to rephrase your message."}

        # Get similar FAQ entries
        docs = vector_store.similarity_search_with_score(question, k=1)
        
        if docs:
            doc, score = docs[0]
            try:
                content = json.loads(doc.page_content)
                faq_question = content.get("question", "")
                similarity_score = calculate_similarity_score(question, faq_question)
                
                # If similarity is high (>0.9), return FAQ answer directly
                if similarity_score > 0.9:
                    return {"answer": content.get("answer", doc.page_content.strip())}
                    
                # If similarity is lower, generate contextual response using LLM
                else:
                    template = """
                    You are the MRPH customer service assistant. Analyze the user's question and provide a specific, relevant response:

                    User Question: {question}
                    Related FAQ: {faq_content}

                    Instructions:
                    1. Your response must be concise and short with proper meaning. 
                    One complete sentence that addresses all user needs without cutting off.You do not finish with incomplete sentence. Finish in one sentence with proper meaning. No explnation. Direct answer which provides the solutiona or suggestion.
                    2. Respond in ONE SENTENCE only
                    3. Use commas and conjunctions to combine multiple service recommendations
                    4. Be direct and specific about which service to book
                    5. Start with action words like "Book", "Schedule", "Use" when recommending services
                    6. No extra explanations or pleasantries
                    7. For new users → Explain registration process
                    8. For service issues → Match to specific service:
                       - Electrical problems → electrical service
                       - HVAC/cooling/heating → HVAC service
                       - Plumbing/water → plumbing service
                       - Security systems → security service
                       - Power issues → power solutions
                    9. For general inquiries → Explain relevant MRPH feature
                    10. Keep response to few words if possible and if it is not possible to answer the query in few words then use 1 sentence maximum
                    11. Be specific and action-oriented
                    12. Don't default to electrical service

                    Response should focus on the user's specific need or question.
                    """
                    
                    prompt = PromptTemplate(
                        template=template,
                        input_variables=["question", "faq_content"]
                    )
                    
                    response = llm(prompt.format(
                        question=question,
                        faq_content=doc.page_content
                    ))
                    
                    return {"answer": response.strip()}
                    
            except json.JSONDecodeError:
                return {"answer": doc.page_content.strip()}
        
        return {"answer": "I'm sorry, I couldn't find relevant information for that query."}
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing your request: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",  # Changed from main:app to app:app
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )