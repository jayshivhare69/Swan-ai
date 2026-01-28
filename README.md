```python
"""
Companion AI Chatbot with Image 
import os
from typing import Optional
import requests
import json
from datetime import datetime

class CompanionAI:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Companion AI
        
        Args:
            api_key: API key for AI services (Hugging Face, OpenAI, etc.)
        """
        self.api_key = api_key or os.getenv('AI_API_KEY')
        self.conversation_history = []
        self.user_name = None
        self.session_start = datetime.now()
        
    def set_user_name(self, name: str):
        """Set the user's name for personalization"""
        self.user_name = name
        
    def generate_response(self, user_input: str) -> str:
        """
        Generate a conversational response using AI
        
        Args:
            user_input: The user's message
            
        Returns:
            AI-generated response
        """
        # Add user input to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Create a friendly, empathetic system prompt
        system_prompt = """You are a friendly, empathetic AI companion designed to help people 
        feel less lonely. You should:
        - Be warm, caring, and supportive
        - Ask thoughtful questions about the user's day and feelings
        - Share interesting facts or stories when appropriate
        - Be a good listener and respond thoughtfully
        - Encourage positive thinking
        - Remember context from the conversation
        - Be conversational and natural, not robotic"""
        
        try:
            # Option 1: Using Hugging Face Inference API (Free tier available)
            response = self._call_huggingface_api(user_input, system_prompt)
            
        except Exception as e:
            # Fallback to rule-based responses if API fails
            response = self._fallback_response(user_input)
        
        # Add AI response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    def _call_huggingface_api(self, user_input: str, system_prompt: str) -> str:
        """
        Call Hugging Face API for text generation
        Uses Meta's Llama model
        """
        API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Format conversation for Llama
        conversation_text = f"{system_prompt}\n\n"
        for msg in self.conversation_history[-5:]:  # Last 5 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"
        conversation_text += f"User: {user_input}\nAssistant:"
        
        payload = {
            "inputs": conversation_text,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                # Extract only the new assistant response
                assistant_response = generated_text.split("Assistant:")[-1].strip()
                return assistant_response
        
        raise Exception("API call failed")
    
    def _fallback_response(self, user_input: str) -> str:
        """
        Fallback responses when API is unavailable
        """
        user_input_lower = user_input.lower()
        
        # Greeting responses
        if any(greet in user_input_lower for greet in ['hello', 'hi', 'hey', 'greetings']):
            responses = [
                f"Hello{' ' + self.user_name if self.user_name else ''}! It's great to chat with you. How are you doing today?",
                f"Hi there{' ' + self.user_name if self.user_name else ''}! I'm here to keep you company. What's on your mind?",
                "Hey! I'm so glad you're here. How has your day been?"
            ]
            return responses[len(self.conversation_history) % len(responses)]
        
        # Emotional support
        elif any(word in user_input_lower for word in ['sad', 'lonely', 'alone', 'depressed', 'down']):
            return "I'm really sorry you're feeling this way. Remember that it's okay to feel down sometimes - you're not alone in this. Would you like to talk about what's bothering you? Sometimes sharing helps."
        
        elif any(word in user_input_lower for word in ['happy', 'good', 'great', 'wonderful']):
            return "That's wonderful to hear! I'm so glad you're feeling good. What's been making you happy today? I'd love to hear about it!"
        
        # How are you
        elif 'how are you' in user_input_lower:
            return "I'm doing well, thank you for asking! I'm here and ready to chat with you. More importantly, how are YOU doing?"
        
        # Goodbye
        elif any(word in user_input_lower for word in ['bye', 'goodbye', 'see you', 'talk later']):
            return "It was lovely chatting with you! Remember, I'm always here whenever you want to talk. Take care and hope to see you soon! 😊"
        
        # Default response
        else:
            return "That's interesting! Tell me more about that. I'm here to listen and chat with you about anything you'd like."
    
    def generate_image(self, prompt: str, save_path: str = "generated_image.png") -> str:
        """
        Generate an image using AI image generation
        
        Args:
            prompt: Description of the image to generate
            save_path: Where to save the generated image
            
        Returns:
            Path to the generated image or error message
        """
        try:
            # Using Hugging Face Stable Diffusion API
            API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": "blurry, bad quality, distorted",
                    "num_inference_steps": 30
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                # Save the image
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return f"Image generated successfully! Saved as {save_path}"
            else:
                return f"Error generating image: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error generating image: {str(e)}"
    
    def get_conversation_summary(self) -> dict:
        """Get a summary of the current conversation session"""
        return {
            "session_duration": str(datetime.now() - self.session_start),
            "message_count": len(self.conversation_history),
            "user_name": self.user_name
        }
    
    def save_conversation(self, filename: str = "conversation_history.json"):
        """Save conversation history to a file"""
        with open(filename, 'w') as f:
            json.dump({
                "user_name": self.user_name,
                "session_start": self.session_start.isoformat(),
                "conversation": self.conversation_history
            }, f, indent=2)
        return f"Conversation saved to {filename}"


def main():
    """
    Main function to run the Companion AI chatbot
    """
    print("=" * 60)
    print("🤖 Welcome to Companion AI - Your Friendly Chat Buddy! 🤖")
    print("=" * 60)
    print("\nI'm here to chat with you and keep you company!")
    print("I can also generate images for you using AI.")
    print("\nCommands:")
    print("  - Type your message to chat")
    print("  - Type '/image <description>' to generate an image")
    print("  - Type '/bye' to exit")
    print("  - Type '/save' to save conversation")
    print("=" * 60)
    
    # Get API key (you'll need to get one from Hugging Face)
    api_key = input("\nEnter your Hugging Face API key (or press Enter to use fallback mode): ").strip()
    if not api_key:
        print("\n⚠️  Running in fallback mode (limited responses)")
        print("To get full AI capabilities, sign up at https://huggingface.co/")
        api_key = None
    
    # Initialize the AI
    ai = CompanionAI(api_key=api_key)
    
    # Get user's name for personalization
    user_name = input("\nWhat's your name? (optional): ").strip()
    if user_name:
        ai.set_user_name(user_name)
        print(f"\nNice to meet you, {user_name}! 😊")
    
    print("\nLet's start chatting! I'm all ears.\n")
    
    # Main conversation loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['/bye', '/exit', '/quit']:
                summary = ai.get_conversation_summary()
                print(f"\n👋 Thanks for chatting! We talked for {summary['session_duration']}")
                print(f"Total messages: {summary['message_count']}")
                save = input("Would you like to save this conversation? (yes/no): ").lower()
                if save in ['yes', 'y']:
                    filename = ai.save_conversation()
                    print(filename)
                print("Take care! Come back anytime you want to chat! 💙")
                break
            
            elif user_input.lower() == '/save':
                filename = ai.save_conversation()
                print(f"✅ {filename}")
                continue
            
            elif user_input.lower().startswith('/image '):
                prompt = user_input[7:].strip()
                if prompt:
                    print("\n🎨 Generating your image... This may take a moment...")
                    result = ai.generate_image(prompt)
                    print(f"✅ {result}\n")
                else:
                    print("❌ Please provide a description for the image.\n")
                continue
            
            # Generate AI response
            print("\n🤖 ", end="", flush=True)
            response = ai.generate_response(user_input)
            print(f"{response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Take care!")
            break
        except Exception as e:
            print(f"\n❌ Oops, something went wrong: {str(e)}\n")
            continue


if __name__ == "__main__":
    main()
``````python
"""
Companion AI Chatbot with Image Generation
A friendly AI assistant to chat with users and generate images
"""

import os
from typing import Optional
import requests
import json
from datetime import datetime

class CompanionAI:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Companion AI
        
        Args:
            api_key: API key for AI services (Hugging Face, OpenAI, etc.)
        """
        self.api_key = api_key or os.getenv('AI_API_KEY')
        self.conversation_history = []
        self.user_name = None
        self.session_start = datetime.now()
        
    def set_user_name(self, name: str):
        """Set the user's name for personalization"""
        self.user_name = name
        
    def generate_response(self, user_input: str) -> str:
        """
        Generate a conversational response using AI
        
        Args:
            user_input: The user's message
            
        Returns:
            AI-generated response
        """
        # Add user input to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Create a friendly, empathetic system prompt
        system_prompt = """You are a friendly, empathetic AI companion designed to help people 
        feel less lonely. You should:
        - Be warm, caring, and supportive
        - Ask thoughtful questions about the user's day and feelings
        - Share interesting facts or stories when appropriate
        - Be a good listener and respond thoughtfully
        - Encourage positive thinking
        - Remember context from the conversation
        - Be conversational and natural, not robotic"""
        
        try:
            # Option 1: Using Hugging Face Inference API (Free tier available)
            response = self._call_huggingface_api(user_input, system_prompt)
            
        except Exception as e:
            # Fallback to rule-based responses if API fails
            response = self._fallback_response(user_input)
        
        # Add AI response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    def _call_huggingface_api(self, user_input: str, system_prompt: str) -> str:
        """
        Call Hugging Face API for text generation
        Uses Meta's Llama model
        """
        API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Format conversation for Llama
        conversation_text = f"{system_prompt}\n\n"
        for msg in self.conversation_history[-5:]:  # Last 5 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"
        conversation_text += f"User: {user_input}\nAssistant:"
        
        payload = {
            "inputs": conversation_text,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                # Extract only the new assistant response
                assistant_response = generated_text.split("Assistant:")[-1].strip()
                return assistant_response
        
        raise Exception("API call failed")
    
    def _fallback_response(self, user_input: str) -> str:
        """
        Fallback responses when API is unavailable
        """
        user_input_lower = user_input.lower()
        
        # Greeting responses
        if any(greet in user_input_lower for greet in ['hello', 'hi', 'hey', 'greetings']):
            responses = [
                f"Hello{' ' + self.user_name if self.user_name else ''}! It's great to chat with you. How are you doing today?",
                f"Hi there{' ' + self.user_name if self.user_name else ''}! I'm here to keep you company. What's on your mind?",
                "Hey! I'm so glad you're here. How has your day been?"
            ]
            return responses[len(self.conversation_history) % len(responses)]
        
        # Emotional support
        elif any(word in user_input_lower for word in ['sad', 'lonely', 'alone', 'depressed', 'down']):
            return "I'm really sorry you're feeling this way. Remember that it's okay to feel down sometimes - you're not alone in this. Would you like to talk about what's bothering you? Sometimes sharing helps."
        
        elif any(word in user_input_lower for word in ['happy', 'good', 'great', 'wonderful']):
            return "That's wonderful to hear! I'm so glad you're feeling good. What's been making you happy today? I'd love to hear about it!"
        
        # How are you
        elif 'how are you' in user_input_lower:
            return "I'm doing well, thank you for asking! I'm here and ready to chat with you. More importantly, how are YOU doing?"
        
        # Goodbye
        elif any(word in user_input_lower for word in ['bye', 'goodbye', 'see you', 'talk later']):
            return "It was lovely chatting with you! Remember, I'm always here whenever you want to talk. Take care and hope to see you soon! 😊"
        
        # Default response
        else:
            return "That's interesting! Tell me more about that. I'm here to listen and chat with you about anything you'd like."
    
    def generate_image(self, prompt: str, save_path: str = "generated_image.png") -> str:
        """
        Generate an image using AI image generation
        
        Args:
            prompt: Description of the image to generate
            save_path: Where to save the generated image
            
        Returns:
            Path to the generated image or error message
        """
        try:
            # Using Hugging Face Stable Diffusion API
            API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": "blurry, bad quality, distorted",
                    "num_inference_steps": 30
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                # Save the image
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return f"Image generated successfully! Saved as {save_path}"
            else:
                return f"Error generating image: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error generating image: {str(e)}"
    
    def get_conversation_summary(self) -> dict:
        """Get a summary of the current conversation session"""
        return {
            "session_duration": str(datetime.now() - self.session_start),
            "message_count": len(self.conversation_history),
            "user_name": self.user_name
        }
    
    def save_conversation(self, filename: str = "conversation_history.json"):
        """Save conversation history to a file"""
        with open(filename, 'w') as f:
            json.dump({
                "user_name": self.user_name,
                "session_start": self.session_start.isoformat(),
                "conversation": self.conversation_history
            }, f, indent=2)
        return f"Conversation saved to {filename}"


def main():
    """
    Main function to run the Companion AI chatbot
    """
    print("=" * 60)
    print("🤖 Welcome to Companion AI - Your Friendly Chat Buddy! 🤖")
    print("=" * 60)
    print("\nI'm here to chat with you and keep you company!")
    print("I can also generate images for you using AI.")
    print("\nCommands:")
    print("  - Type your message to chat")
    print("  - Type '/image <description>' to generate an image")
    print("  - Type '/bye' to exit")
    print("  - Type '/save' to save conversation")
    print("=" * 60)
    
    # Get API key (you'll need to get one from Hugging Face)
    api_key = input("\nEnter your Hugging Face API key (or press Enter to use fallback mode): ").strip()
    if not api_key:
        print("\n⚠️  Running in fallback mode (limited responses)")
        print("To get full AI capabilities, sign up at https://huggingface.co/")
        api_key = None
    
    # Initialize the AI
    ai = CompanionAI(api_key=api_key)
    
    # Get user's name for personalization
    user_name = input("\nWhat's your name? (optional): ").strip()
    if user_name:
        ai.set_user_name(user_name)
        print(f"\nNice to meet you, {user_name}! 😊")
    
    print("\nLet's start chatting! I'm all ears.\n")
    
    # Main conversation loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['/bye', '/exit', '/quit']:
                summary = ai.get_conversation_summary()
                print(f"\n👋 Thanks for chatting! We talked for {summary['session_duration']}")
                print(f"Total messages: {summary['message_count']}")
                save = input("Would you like to save this conversation? (yes/no): ").lower()
                if save in ['yes', 'y']:
                    filename = ai.save_conversation()
                    print(filename)
                print("Take care! Come back anytime you want to chat! 💙")
                break
            
            elif user_input.lower() == '/save':
                filename = ai.save_conversation()
                print(f"✅ {filename}")
                continue
            
            elif user_input.lower().startswith('/image '):
                prompt = user_input[7:].strip()
                if prompt:
                    print("\n🎨 Generating your image... This may take a moment...")
                    result = ai.generate_image(prompt)
                    print(f"✅ {result}\n")
                else:
                    print("❌ Please provide a description for the image.\n")
                continue
            
            # Generate AI response
            print("\n🤖 ", end="", flush=True)
            response = ai.generate_response(user_input)
            print(f"{response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Take care!")
            break
        except Exception as e:
            print(f"\n❌ Oops, something went wrong: {str(e)}\n")
            continue


if __name__ == "__main__":
    main()
```
