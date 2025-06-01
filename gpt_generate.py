import json
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from huggingface_hub import login, hf_hub_download
import torch
from tqdm import tqdm
import argparse
import time
import random
from functools import lru_cache
import heapq
from dotenv import load_dotenv

load_dotenv()

# Set logging level to show download progress
logging.set_verbosity_info()


# Add function to handle HF login
def setup_hf_auth():
    """Setup HuggingFace authentication"""
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("Please enter your HuggingFace token (from https://huggingface.co/settings/tokens):")
        token = input().strip()
        os.environ['HF_TOKEN'] = token
    try:
        login(token=token)
        print("Successfully authenticated with HuggingFace")
    except Exception as e:
        raise RuntimeError(f"Authentication failed: {str(e)}")

@lru_cache(maxsize=None)
def download_and_load_mixtral(models_dir):
    """
    Load Mistral model from local directory or download if not available.
    """
    model_name = "mistralai/Mistral-7B-v0.1"
    local_model_path = os.path.join(models_dir, "Mistral-7B-v0.1")
    
    print(f"Attempting to load Mistral model...")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    # Ensure HF authentication is set up
    setup_hf_auth()
    
    try:
        if os.path.exists(local_model_path):
            print(f"Loading from local path: {local_model_path}")
            
            # Check if model files exist in the local path
            required_model_files_found = False
            
            # Check for safetensors files (Mistral uses sharded safetensors)
            safetensors_pattern = os.path.join(local_model_path, "model-*.safetensors")
            import glob
            safetensors_files = glob.glob(safetensors_pattern)
            if safetensors_files:
                print(f"Found {len(safetensors_files)} safetensors model files")
                required_model_files_found = True
            else:
                # Check for other model file formats
                model_file_options = [
                    "pytorch_model.bin",
                    "tf_model.h5", 
                    "model.ckpt.index",
                    "flax_model.msgpack"
                ]
                for model_file in model_file_options:
                    if os.path.exists(os.path.join(local_model_path, model_file)):
                        print(f"Found model file: {model_file}")
                        required_model_files_found = True
                        break
                
                if not required_model_files_found:
                    print("No model files found in local directory. Will download model files.")
                    # We'll continue and let the model loading handle the download
            
            # Check if tokenizer files exist in the local path
            required_tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]
            missing_files = [f for f in required_tokenizer_files if not os.path.exists(os.path.join(local_model_path, f))]
            
            if missing_files:
                print(f"Tokenizer files missing in local directory: {', '.join(missing_files)}")
                print("Downloading tokenizer from HuggingFace...")
                
                try:
                    # First try to download the complete tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        token=os.getenv('HF_TOKEN'),
                        use_fast=False,
                        legacy=True
                    )
                    # Save tokenizer to local path
                    print(f"Saving tokenizer to {local_model_path}")
                    tokenizer.save_pretrained(local_model_path)
                except Exception as tokenizer_error:
                    print(f"Error downloading tokenizer: {str(tokenizer_error)}")
                    print("Attempting to download individual tokenizer files...")
                    
                    # Try direct download of individual tokenizer files
                    from huggingface_hub import hf_hub_download
                    import shutil
                    
                    for file in required_tokenizer_files:
                        try:
                            file_path = hf_hub_download(
                                repo_id=model_name,
                                filename=file,
                                token=os.getenv('HF_TOKEN')
                            )
                            # Copy the file to the local model path
                            shutil.copy(file_path, os.path.join(local_model_path, file))
                            print(f"Downloaded {file} successfully")
                        except Exception as file_error:
                            print(f"Error downloading {file}: {str(file_error)}")
                    
                    # Try loading again after downloading individual files
                    print("Attempting to load tokenizer with downloaded files...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        local_model_path,
                        use_fast=False,
                        legacy=True
                    )
            else:
                # Force use of slow tokenizer to avoid tokenizer_fast issues
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        local_model_path,
                        use_fast=False,
                        legacy=True
                    )
                except Exception as tokenizer_error:
                    print(f"Error loading tokenizer from local path: {str(tokenizer_error)}")
                    print("Attempting to download tokenizer directly from HuggingFace...")
                    
                    # Try downloading directly from HuggingFace as a fallback
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        token=os.getenv('HF_TOKEN'),
                        use_fast=False,
                        legacy=True
                    )
                    
                    # Save the tokenizer again
                    print(f"Re-saving tokenizer to {local_model_path}")
                    tokenizer.save_pretrained(local_model_path)
                
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            print("Loading model weights (this may take a few minutes)...")
            # Adjust loading parameters based on CUDA availability
            load_params = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "use_safetensors": True,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }
            
            # Only add 8-bit quantization if CUDA is available
            if cuda_available:
                load_params.update({
                    "load_in_8bit": True,
                    "offload_folder": "offload",
                    "offload_state_dict": True
                })
            else:
                # For CPU, we need to be more conservative with memory usage
                # and avoid quantization which requires GPU
                load_params.update({
                    "device_map": {"":"cpu"},
                    "torch_dtype": torch.float32  # Use float32 on CPU for better compatibility
                })
            
            try:
                # If we didn't find model files earlier, force download from HF
                if not required_model_files_found:
                    print("Downloading model files from HuggingFace...")
                    # Add token to load_params for authentication
                    download_params = load_params.copy()
                    download_params["token"] = os.getenv('HF_TOKEN')
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,  # Use remote model name to force download
                        **download_params
                    )
                    print("Model files downloaded successfully")
                    
                    # Save the model to local path
                    try:
                        print(f"Saving model to {local_model_path}")
                        model.save_pretrained(local_model_path, safe_serialization=True)
                        print("Model saved successfully")
                    except Exception as save_error:
                        print(f"Warning: Could not save model locally: {str(save_error)}")
                        print("Will continue using the downloaded model")
                else:
                    # Try to load from local path as model files were found
                    model = AutoModelForCausalLM.from_pretrained(
                        local_model_path,
                        **load_params
                    )
            except Exception as model_error:
                print(f"\nError loading local model with current settings: {str(model_error)}")
                print("Attempting to load with minimal settings...")
                
                # Fallback to minimal settings
                minimal_params = {
                    "device_map": "cpu",
                    "torch_dtype": torch.float32,
                    "low_cpu_mem_usage": True
                }
                
                try:
                    # Try loading with minimal settings from local path first
                    model = AutoModelForCausalLM.from_pretrained(
                        local_model_path,
                        **minimal_params
                    )
                    print("Successfully loaded model with minimal settings.")
                except Exception as local_fallback_error:
                    print(f"\nLocal fallback loading failed: {str(local_fallback_error)}")
                    print("Attempting to download model directly from HuggingFace...")
                    
                    try:
                        # Add token for authentication
                        minimal_params["token"] = os.getenv('HF_TOKEN')
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,  # Use remote model name
                            **minimal_params
                        )
                        print("Successfully downloaded and loaded model from HuggingFace.")
                    except Exception as remote_fallback_error:
                        print(f"\nRemote fallback loading also failed: {str(remote_fallback_error)}")
                        raise
        else:
            # Directory doesn't exist, create it and download everything
            os.makedirs(local_model_path, exist_ok=True)
            print(f"Downloading from HuggingFace: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=os.getenv('HF_TOKEN'),
                use_fast=False,
                legacy=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print("\nDownloading model weights (this may take 10-15 minutes)...")
            print("The model is large (~7GB), please be patient...")
            
            # Adjust loading parameters based on CUDA availability
            load_params = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "use_safetensors": True,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "token": os.getenv('HF_TOKEN')
            }
            
            # Only add 8-bit quantization if CUDA is available
            if cuda_available:
                load_params.update({
                    "load_in_8bit": True,
                    "offload_folder": "offload",
                    "offload_state_dict": True
                })
            else:
                # For CPU, we need to be more conservative with memory usage
                # and avoid quantization which requires GPU
                load_params.update({
                    "device_map": {"":"cpu"},
                    "torch_dtype": torch.float32  # Use float32 on CPU for better compatibility
                })
            
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **load_params
                )
            except Exception as model_error:
                print(f"\nError loading model with current settings: {str(model_error)}")
                print("Attempting to load with minimal settings...")
                
                # Fallback to minimal settings
                minimal_params = {
                    "device_map": "cpu",
                    "torch_dtype": torch.float32,
                    "low_cpu_mem_usage": True,
                    "token": os.getenv('HF_TOKEN')
                }
                
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        **minimal_params
                    )
                    print("Successfully loaded model with minimal settings.")
                except Exception as fallback_error:
                    print(f"\nFallback loading also failed: {str(fallback_error)}")
                    raise

            # Skip saving the model as it can cause issues with meta tensors
            # Just save the tokenizer which is smaller and less problematic
            os.makedirs(local_model_path, exist_ok=True)
            print(f"\nSaving tokenizer to {local_model_path}")
            try:
                tokenizer.save_pretrained(local_model_path)
                # Verify tokenizer files were saved correctly
                if not (os.path.exists(os.path.join(local_model_path, "tokenizer_config.json")) and 
                        os.path.exists(os.path.join(local_model_path, "tokenizer.json"))):
                    print("Warning: Tokenizer files may not have been saved correctly. Will attempt to fix...")
                    # Force download of specific tokenizer files
                    from huggingface_hub import hf_hub_download
                    for file in ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]:
                        try:
                            file_path = hf_hub_download(
                                repo_id=model_name,
                                filename=file,
                                token=os.getenv('HF_TOKEN')
                            )
                            # Copy the file to the local model path
                            import shutil
                            shutil.copy(file_path, os.path.join(local_model_path, file))
                            print(f"Downloaded {file} successfully")
                        except Exception as file_error:
                            print(f"Error downloading {file}: {str(file_error)}")
            except Exception as save_error:
                print(f"Error saving tokenizer: {str(save_error)}. Will attempt direct download.")
                # Try direct download of tokenizer files
                from huggingface_hub import hf_hub_download
                for file in ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]:
                    try:
                        file_path = hf_hub_download(
                            repo_id=model_name,
                            filename=file,
                            token=os.getenv('HF_TOKEN')
                        )
                        # Copy the file to the local model path
                        import shutil
                        shutil.copy(file_path, os.path.join(local_model_path, file))
                        print(f"Downloaded {file} successfully")
                    except Exception as file_error:
                        print(f"Error downloading {file}: {str(file_error)}")
            
            print("Note: Model weights will not be saved locally to avoid tensor conversion errors.")

        config = model.config.to_dict()
        return config, model, tokenizer
        
    except Exception as e:
        error_msg = str(e)
        print(f"\nError loading model: {error_msg}")
        print("\nTroubleshooting steps:")
        
        # Provide specific advice based on the error message
        if "no file named pytorch_model.bin" in error_msg.lower() or "model-*.safetensors" in error_msg.lower():
            print("Model files missing troubleshooting:")
            print("1. The model files are missing in your local directory. Try deleting the local directory and run again to download fresh files:")
            print(f"   rm -rf {local_model_path}")
            print("2. Ensure you have at least 15GB of free disk space for the model files")
            print("3. Try downloading the model files manually from HuggingFace:")
            print(f"   https://huggingface.co/{model_name}/tree/main")
            print("4. If you have limited disk space, consider using a smaller model")
        elif "tokenizer" in error_msg.lower():
            print("Tokenizer-specific troubleshooting:")
            print("1. Delete the local model directory and try again to download fresh files")
            print(f"   rm -rf {local_model_path}")
            print("2. Ensure you have proper permissions to write to the model directory")
            print("3. Try downloading the tokenizer files manually from HuggingFace")
            print(f"   https://huggingface.co/{model_name}/tree/main")
        
        print("\nGeneral troubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have at least 16GB of free RAM")
        print("3. Check if you have accepted the model terms at:")
        print(f"   https://huggingface.co/{model_name}")
        print("4. Verify your HF_TOKEN has the correct permissions")
        print("5. Try clearing the HuggingFace cache:")
        print("   ~/.cache/huggingface/hub")
        print("6. If using CPU, the model may be too large. Consider using a GPU.")
        
        raise RuntimeError(f"Failed to load model: {error_msg}")


class GPT2Chatbot:
    def __init__(self, models_dir="mixtral", use_fp16=True):
        self.models_dir = models_dir
        self.use_fp16 = use_fp16
        self.conversation_history = []
        self.max_history_tokens = 4096  # Increased for Mixtral's context window
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Improved persona examples - more focused and specific to guide the model better
        self.initial_examples = [
            {"user": "Hello, how are you?", 
             "bot": "I'm doing great! I'm here to chat and help with any questions you might have. How's your day going?"},
            {"user": "What can you do?", 
             "bot": "I can have conversations about many topics! I'm good at discussing science, history, philosophy, or just casual chats. I can also help with creative writing or answering questions. What would you like to talk about today?"},
            {"user": "Tell me something interesting", 
             "bot": "Did you know that octopuses have nine brains? They have a central brain and then eight additional mini-brains - one in each arm! This distributed nervous system allows each arm to act somewhat independently. That's why octopuses are so good at solving puzzles and navigating complex environments."}
        ]
        
        # Response length controls
        self.response_length_presets = {
            "brief": 30,
            "normal": 60,
            "detailed": 120,
            "conversational": 50
        }
        self.current_response_length = "normal"
        
        print("Initializing Mixtral Chatbot...")
        self.setup_model()
        
        # Add topic-specific prompting templates
        self.topic_templates = {
            "science": "You are discussing science topics with great knowledge and clarity. Your responses are accurate, educational and engaging.",
            "philosophy": "You are a thoughtful philosophy tutor who asks socratic questions and provides deep insights.",
            "casual": "You are having a friendly, casual conversation. Your responses are warm, personable and conversational."
        }
        self.active_template = None
        
        # Cache for repeated queries
        self.response_cache = {}
        self.cache_size = 50  # Maximum number of cached responses
        
    def setup_model(self):
        """Initialize and set up the Mixtral model."""
        print(f"Loading model on {self.device}...")
        start_time = time.time()
        
        # Declare global tokenizer first
        global tokenizer
        
        # Download and load model
        self.config, self.model, tokenizer = download_and_load_mixtral(self.models_dir)
        
        # No need for manual weight loading as transformers handles it
        self.model.eval()
        
        # Report loading time
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Initialize conversation history
        self.conversation_history = self.initial_examples.copy()
        
        # Fallback responses for error cases
        self.fallback_responses = [
            "That's an interesting point. Could you tell me more about what you're thinking?",
            "I'm curious to hear your thoughts on this topic. What aspects interest you most?",
            "That's a good question. Let me think about how to approach this...",
            "I'd love to explore that idea more with you. What specifically caught your interest?",
            "That's worth considering from multiple angles. What's your perspective on it?",
        ]
        
        print("Mixtral Chatbot ready!")
    
    def format_prompt(self, user_input):
        """Format the conversation history into a prompt that guides the model's responses."""
        # Start with a concise system-style prompt to guide the model
        if self.active_template:
            formatted_history = f"{self.topic_templates[self.active_template]}\n\n"
        else:
            formatted_history = (
                "The following is a helpful, thoughtful conversation. "
                "The assistant provides concise, accurate responses.\n\n"
            )
        
        # Calculate how many history turns we can include
        # Add the most recent history, keeping it concise
        max_history_turns = min(5, len(self.conversation_history))
        for entry in self.conversation_history[-max_history_turns:]:
            formatted_history += f"Human: {entry['user']}\nAssistant: {entry['bot']}\n\n"
            
        # Add current user input
        formatted_prompt = formatted_history + f"Human: {user_input}\nAssistant:"
        return formatted_prompt
        
    def cache_key(self, user_input, temperature, max_tokens):
        """Generate a cache key for a given query and parameters."""
        # Simple hashing function for cache key
        history_str = str([h['user'] + h['bot'] for h in self.conversation_history[-2:]])
        return f"{user_input}_{temperature}_{max_tokens}_{hash(history_str)}"
        
    def chat(self, user_input, max_tokens=None, temperature=0.7, top_p=0.9):
        """Generate a response to the user input."""
        # Check cache first
        cache_key = self.cache_key(user_input, temperature, max_tokens)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Determine response length based on preset if not specified
        if max_tokens is None:
            max_tokens = self.response_length_presets[self.current_response_length]
            
        # Format prompt with conversation history
        prompt = self.format_prompt(user_input)
        
        # Convert to token IDs with attention mask
        tokenizer_output = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        ).to(self.device)
        
        input_ids = tokenizer_output.input_ids
        attention_mask = tokenizer_output.attention_mask
        
        # Check if we need to truncate history (if prompt is too long)
        if input_ids.shape[1] > self.max_history_tokens:
            # Keep fewer conversation turns to fit in context window
            self.conversation_history = (
                self.initial_examples[:1] +  # Keep one guiding example
                self.conversation_history[-3:]  # Keep only most recent turns
            )
            prompt = self.format_prompt(user_input)
            tokenizer_output = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            ).to(self.device)
            input_ids = tokenizer_output.input_ids
            attention_mask = tokenizer_output.attention_mask
        
        # Generate response with improved function and better stop phrases
        stop_phrases = ["\nHuman:", "\n\nHuman:", "\nUser:", "User:", "\n\n"]
        
        try:
            # Measure generation time
            start_time = time.time()
            
            output_ids, response_text = generate(
                model=self.model,
                idx=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                context_size=self.config["max_position_embeddings"],
                temperature=temperature,
                top_k=40,
                top_p=top_p,
                stop_phrases=stop_phrases,
                min_tokens=5
            )
            
            # Calculate performance metrics
            generation_time = time.time() - start_time
            tokens_per_second = len(response_text.split()) / max(0.1, generation_time)
            
            # Clean up the response
            response_text = self._clean_response(response_text)
            
            # Update conversation history
            self.conversation_history.append({"user": user_input, "bot": response_text})
            
            # Keep history at a reasonable size
            if len(self.conversation_history) > 10:
                # Keep introduction examples + recent conversation
                self.conversation_history = (
                    self.initial_examples[:1] +  # Keep one guiding example
                    self.conversation_history[-8:]  # Keep recent conversation
                )
            
            # Cache the response
            if len(self.response_cache) > self.cache_size:
                # Remove a random key if cache is full
                old_key = random.choice(list(self.response_cache.keys()))
                del self.response_cache[old_key]
            self.response_cache[cache_key] = response_text
            
            return response_text
            
        except Exception as e:
            print(f"Error in generation: {str(e)}")
            return random.choice(self.fallback_responses)
    
    def _clean_response(self, text):
        """Clean up the generated response text."""
        # Remove common prefixes
        prefixes = ["AI:", "Bot:", "Assistant:", "GPT:", "GPT-2:"]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                
        # Remove surrounding quotes if present
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()
            
        # Ensure the text is not too short
        if len(text.strip()) < 10:
            text = random.choice(self.fallback_responses)
            
        return text.strip()
    
    def set_response_length(self, length_preset):
        """Set the response length preset."""
        if length_preset in self.response_length_presets:
            self.current_response_length = length_preset
            return f"Response length set to {length_preset}"
        else:
            return f"Invalid preset. Available options: {', '.join(self.response_length_presets.keys())}"
    
    def set_topic(self, topic):
        """Set a specific topic focus to guide responses."""
        if topic.lower() in self.topic_templates:
            self.active_template = topic.lower()
            return f"Topic set to {topic}. Responses will be tailored accordingly."
        elif topic.lower() == "default" or topic.lower() == "none":
            self.active_template = None
            return "Topic focus has been reset to default."
        else:
            return f"Topic not recognized. Available topics: {', '.join(self.topic_templates.keys())}"
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = self.initial_examples.copy()
        self.active_template = None
        return "Conversation history has been reset."


@torch.no_grad()
def generate(model, idx, attention_mask, max_new_tokens, context_size, temperature=0.8, 
            top_k=40, top_p=0.9, stop_phrases=None, min_tokens=5):
    """
    Adapted generation function for Mixtral model with proper attention mask
    """
    model.eval()
    
    # Get the input shape
    input_len = idx.shape[1]
    
    # Generate with proper attention mask
    outputs = model.generate(
        idx,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        use_cache=True
    )
    
    # Get generated text
    generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    # Handle stop phrases
    if stop_phrases:
        for phrase in stop_phrases:
            if phrase in generated_text:
                generated_text = generated_text[:generated_text.index(phrase)]
    
    return outputs, generated_text


def main():
    parser = argparse.ArgumentParser(description="Optimized Mixtral Chatbot")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for text generation (0.0-1.0)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling parameter (0.0-1.0)")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Maximum number of tokens to generate (overrides length preset)")
    parser.add_argument("--length", type=str, default="normal",
                        choices=["brief", "normal", "detailed", "conversational"],
                        help="Response length preset")
    parser.add_argument("--topic", type=str, default=None,
                        choices=["science", "philosophy", "casual", "none"],
                        help="Set topic-specific response style")
    parser.add_argument("--prime", action="store_true",
                        help="Prime the model with example conversations before use")
    parser.add_argument("--no_fp16", action="store_true",
                        help="Disable FP16 (half-precision) for better compatibility")
    
    args = parser.parse_args()
    
    print("\n=== Mixtral Chatbot ===")
    print("Loading model and preparing for chat...")
    
    # Create and initialize chatbot
    chatbot = GPT2Chatbot(use_fp16=not args.no_fp16)
    chatbot.current_response_length = args.length
    
    if args.topic and args.topic != "none":
        chatbot.set_topic(args.topic)
    
    # Prime the model with a simple exchange to improve future responses
    if args.prime:
        print("Priming the model with example conversations...")
        _ = chatbot.chat("Tell me something interesting about quantum physics", 
                         temperature=args.temperature)
        _ = chatbot.chat("What makes a good conversation?", 
                         temperature=args.temperature)
        chatbot.reset_conversation()
        print("Priming complete.")
    
    # Print welcome message
    print("\n" + "="*50)
    print("Mixtral Chatbot is ready for conversation!")
    print("Commands:")
    print("  exit - Quit the chatbot")
    print("  reset - Clear conversation history")
    print("  temp <value> - Set temperature (0.0-1.0)")
    print("  top_p <value> - Set top-p sampling (0.0-1.0)")
    print(f"  length <preset> - Set response length ({', '.join(chatbot.response_length_presets.keys())})")
    print(f"  topic <topic> - Set topic focus ({', '.join(list(chatbot.topic_templates.keys()) + ['none'])})")
    print("  help - Show this help message")
    print("="*50 + "\n")
    
    # Main chat loop
    current_temp = args.temperature
    current_top_p = args.top_p
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            # Handle special commands
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            elif user_input.lower() == "reset":
                print(chatbot.reset_conversation())
                continue
            elif user_input.lower() == "help":
                print("Commands:")
                print("  exit - Exit the chatbot")
                print("  reset - Clear conversation history")
                print("  temp <value> - Change the temperature (between 0.0 and 1.0)")
                print("  top_p <value> - Change the top-p sampling value (between 0.0 and 1.0)")
                print(f"  length <preset> - Change response length ({', '.join(chatbot.response_length_presets.keys())})")
                print(f"  topic <topic> - Set topic focus ({', '.join(list(chatbot.topic_templates.keys()) + ['none'])})")
                print("  help - Show this help message")
                continue
            elif user_input.lower().startswith("temp "):
                try:
                    new_temp = float(user_input.split(" ")[1])
                    if 0.0 <= new_temp <= 1.0:
                        current_temp = new_temp
                        print(f"Temperature set to {current_temp}")
                    else:
                        print("Temperature must be between 0.0 and 1.0")
                    continue
                except (IndexError, ValueError):
                    print("Invalid format. Use 'temp X' where X is between 0.0 and 1.0")
                    continue
            elif user_input.lower().startswith("top_p "):
                try:
                    new_top_p = float(user_input.split(" ")[1])
                    if 0.0 <= new_top_p <= 1.0:
                        current_top_p = new_top_p
                        print(f"Top-p sampling set to {current_top_p}")
                    else:
                        print("Top-p must be between 0.0 and 1.0")
                    continue
                except (IndexError, ValueError):
                    print("Invalid format. Use 'top_p X' where X is between 0.0 and 1.0")
                    continue
            elif user_input.lower().startswith("length "):
                preset = user_input.split(" ")[1].lower()
                print(chatbot.set_response_length(preset))
                continue
            elif user_input.lower().startswith("topic "):
                topic = user_input.split(" ")[1].lower()
                print(chatbot.set_topic(topic))
                continue
                
            # Generate response
            try:
                start_time = time.time()
                response = chatbot.chat(user_input, 
                                      max_tokens=args.max_tokens, 
                                      temperature=current_temp,
                                      top_p=current_top_p)
                
                generation_time = time.time() - start_time
                tokens = len(response.split())
                speed = tokens / max(0.1, generation_time)
                
                print(f"Bot: {response}")
                print(f"[Generated {tokens} tokens in {generation_time:.2f}s ({speed:.1f} tokens/sec)]")
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                print("Bot: I'm having trouble processing that. Could you try again with different wording?")
        except KeyboardInterrupt:
            print("\nDetected keyboard interrupt. Type 'exit' to quit.")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()