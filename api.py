from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import torch
import json
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import gc

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model and pipelines
model = None
tokenizer = None
sentiment_pipe = None
model_loaded = False

def initialize_models():
    """Initialize AI models on startup"""
    global model, tokenizer, sentiment_pipe, model_loaded
    
    try:
        logger.info("🤖 Initializing AI models...")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Login to Hugging Face
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            login(token=hf_token)
            logger.info("✅ Logged in to Hugging Face")
        
        # Initialize sentiment analysis pipeline with specific model
        sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=False
        )
        logger.info("✅ Sentiment analysis pipeline loaded")
        
        # Load Phi-3 model
        model_path = os.getenv('MODEL_PATH', './models/phi-3')
        
        if os.path.exists(model_path):
            # Check if GPU is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            try:
                # Try loading with quantization only if GPU is available
                if torch.cuda.is_available():
                    logger.info("🔧 Loading model with GPU quantization...")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True
                    )
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        quantization_config=bnb_config,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        attn_implementation="eager"  # Use eager attention
                    )
                    logger.info("✅ Model loaded with GPU quantization")
                else:
                    logger.info("🔧 Loading model on CPU without quantization...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="cpu",
                        torch_dtype=torch.float32,  # Use float32 for CPU
                        trust_remote_code=True,
                        attn_implementation="eager"  # Use eager attention
                    )
                    logger.info("✅ Model loaded on CPU")
                
            except Exception as model_error:
                logger.error(f"❌ Error loading model: {model_error}")
                logger.info("🔄 Trying alternative loading method...")
                
                # Fallback: Load without any special configs
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32 if device == "cpu" else torch.float16,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
                logger.info("✅ Model loaded with fallback method")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                padding_side="left",
                trust_remote_code=True
            )
            
            # Set pad token if not exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model_loaded = True
            logger.info("✅ Phi-3 model and tokenizer loaded successfully")
            
        else:
            logger.warning("⚠️ Model path not found, using fallback responses")
            logger.info(f"Expected model path: {model_path}")
            # List available directories to help debug
            parent_dir = os.path.dirname(model_path)
            if os.path.exists(parent_dir):
                logger.info(f"Available directories in {parent_dir}: {os.listdir(parent_dir)}")
            model_loaded = False
            
    except Exception as e:
        logger.error(f"❌ Error initializing models: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        model_loaded = False

def analyze_sentiment(text):
    """Analyze sentiment of given text"""
    try:
        if sentiment_pipe is None:
            return "NEUTRAL"
        
        result = sentiment_pipe(text)
        sentiment = result[0]['label']
        confidence = result[0]['score']
        
        # Map different sentiment labels to standard format
        sentiment_mapping = {
            'POSITIVE': 'POSITIVE',
            'NEGATIVE': 'NEGATIVE',
            'LABEL_0': 'NEGATIVE',  # Some models use LABEL_0/1
            'LABEL_1': 'POSITIVE',
            'NEUTRAL': 'NEUTRAL'
        }
        
        mapped_sentiment = sentiment_mapping.get(sentiment, 'NEUTRAL')
        
        logger.info(f"Sentiment: {mapped_sentiment} (confidence: {confidence:.2f})")
        return mapped_sentiment
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return "NEUTRAL"

def generate_prompt(sentiment, user_input, conversation_history=None):
    """Generate appropriate prompt based on sentiment and conversation history"""
    
    # Build context from conversation history
    context = ""
    if conversation_history:
        for msg in conversation_history[-4:]:  # Last 4 messages for context
            sender = msg.get('sender', 'user')
            text = msg.get('text', '')
            if sender == 'user':
                context += f"User: {text}\n"
            else:
                context += f"Assistant: {text}\n"
    
    # Sentiment-based prompts
    sentiment_prompts = {
        "POSITIVE": f"""You are a kind, witty AI assistant who responds warmly and engagingly.

{context}User: {user_input}

Generate a friendly, uplifting response that matches the positive mood. Be encouraging and supportive.
Assistant:""",

        "NEGATIVE": f"""You are a supportive, empathetic AI assistant who comforts users with care and understanding.

{context}User: {user_input}

Generate a warm, comforting response that shows empathy and provides gentle encouragement. Be compassionate and understanding.
Assistant:""",

        "NEUTRAL": f"""You are a polite, helpful AI assistant who responds thoughtfully and kindly.

{context}User: {user_input}

Generate a thoughtful, balanced response that is helpful and engaging.
Assistant:"""
    }
    
    return sentiment_prompts.get(sentiment, sentiment_prompts["NEUTRAL"])

def generate_text(prompt):
    """Generate text using the Phi-3 model"""
    try:
        if not model_loaded or model is None or tokenizer is None:
            # Fallback responses when model is not available
            fallback_responses = [
                "I understand what you're saying. Could you tell me more about that?",
                "That's interesting! I'd love to hear more about your thoughts on this.",
                "Thank you for sharing that with me. How are you feeling about it?",
                "I appreciate you opening up. What would you like to explore further?",
                "That sounds important to you. Can you help me understand better?"
            ]
            import random
            return random.choice(fallback_responses)
        
        # Tokenize input
        model_inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Move to appropriate device
        device = next(model.parameters()).device
        model_inputs = {key: val.to(device) for key, val in model_inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=80,
                do_sample=False,
                temperature=0.7,
                top_k=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=False
            )
        
        # Decode response
        full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract only the new generated text
        response = full_output[len(prompt):].strip()
        
        # Clean up response
        if not response:
            response = "I understand. Could you tell me more about that?"
        
        # Limit response length
        if len(response) > 500:
            response = response[:500] + "..."
        
        logger.info(f"Generated response: {response[:100]}...")
        return response
        
    except Exception as e:
        logger.error(f"Error in text generation: {str(e)}")
        return "I'm having trouble processing that right now. Could you please try rephrasing your message?"

# API ENDPOINTS

@app.route('/api/ai/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    gpu_info = None
    if torch.cuda.is_available():
        try:
            gpu_info = {
                'name': torch.cuda.get_device_name(0),
                'total_memory': torch.cuda.get_device_properties(0).total_memory,
                'allocated_memory': torch.cuda.memory_allocated(0),
                'cached_memory': torch.cuda.memory_reserved(0)
            }
        except:
            gpu_info = {'error': 'Unable to get GPU info'}
    
    return jsonify({
        'status': 'OK',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat(),
        'gpu_available': torch.cuda.is_available(),
        'gpu_info': gpu_info,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })

@app.route('/api/ai/generate', methods=['POST'])
def generate_response():
    """Main endpoint for generating AI responses"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        message = data.get('message', '').strip()
        conversation_history = data.get('conversation_history', [])
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Analyze sentiment
        sentiment = analyze_sentiment(message)
        
        # Generate prompt
        prompt = generate_prompt(sentiment, message, conversation_history)
        
        # Generate response
        response = generate_text(prompt)
        
        # Log the interaction
        logger.info(f"User: {message[:50]}...")
        logger.info(f"AI: {response[:50]}...")
        
        return jsonify({
            'response': response,
            'sentiment': sentiment,
            'timestamp': datetime.now().isoformat(),
            'model_used': 'Phi-3' if model_loaded else 'Fallback'
        })
        
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'Something went wrong while generating the response'
        }), 500

@app.route('/api/ai/sentiment', methods=['POST'])
def analyze_sentiment_endpoint():
    """Endpoint for sentiment analysis only"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        sentiment = analyze_sentiment(text)
        
        return jsonify({
            'sentiment': sentiment,
            'text': text,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'Something went wrong during sentiment analysis'
        }), 500

@app.route('/api/ai/model/reload', methods=['POST'])
def reload_models():
    """Endpoint to reload AI models"""
    try:
        global model_loaded
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Reinitialize models
        initialize_models()
        
        return jsonify({
            'message': 'Models reloaded successfully',
            'model_loaded': model_loaded,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error reloading models: {str(e)}")
        return jsonify({
            'error': 'Failed to reload models',
            'message': str(e)
        }), 500

@app.route('/api/ai/stats', methods=['GET'])
def get_stats():
    """Get AI service statistics"""
    try:
        stats = {
            'model_loaded': model_loaded,
            'model_path': os.getenv('MODEL_PATH', './models/phi-3'),
            'gpu_available': torch.cuda.is_available(),
            'timestamp': datetime.now().isoformat()
        }
        
        if torch.cuda.is_available():
            try:
                stats.update({
                    'gpu_name': torch.cuda.get_device_name(0),
                    'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory,
                    'gpu_memory_allocated': torch.cuda.memory_allocated(0),
                    'gpu_memory_cached': torch.cuda.memory_reserved(0)
                })
            except:
                stats['gpu_info'] = 'Unable to get GPU stats'
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({
            'error': 'Failed to get statistics',
            'message': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

if __name__ == '__main__':
    # Initialize models before starting the server
    initialize_models()
    
    # Start server
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Starting Flask AI API on port {port}")
    logger.info(f"🔧 Debug mode: {debug}")
    logger.info(f"🤖 Model loaded: {model_loaded}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )