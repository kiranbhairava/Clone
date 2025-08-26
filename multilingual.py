from typing import Dict, List, Optional
from datetime import datetime

class RomanizedLanguageAdapter:
    """
    Romanized Language Switching - English text, different language content
    Perfect for Indian multilingual scenarios
    """
    
    ROMANIZED_LANGUAGE_INSTRUCTIONS = {
        'english': {
            'instruction': "Respond naturally in English.",
            'style_note': "Use standard English expressions and vocabulary.",
            'example': "Hello! I'm doing great today. How are you?"
        },
        
        'hindi': {
            'instruction': """Respond in Hindi language but write using English letters (Roman script). 
            Use Hindi words, phrases, and expressions but spell them in English letters.""",
            'style_note': "Use Hindi vocabulary in Roman letters. Mix English words naturally when common in Hindi conversations.",
            'example': "Namaste! Main aaj bahut achha feel kar raha hoon. Aap kaise hain?"
        },
        
        'tamil': {
            'instruction': """Respond in Tamil language but write using English letters (Roman script). 
            Use Tamil words, phrases, and expressions but spell them in English letters.""",
            'style_note': "Use Tamil vocabulary in Roman letters. Natural mixing of English words is common.",
            'example': "Vanakkam! Naan inniki romba nalla irukken. Neenga eppadi irukkeenga?"
        },
        
        'telugu': {
            'instruction': """Respond in Telugu language but write using English letters (Roman script). 
            Use Telugu words, phrases, and expressions but spell them in English letters.""",
            'style_note': "Use Telugu vocabulary in Roman letters. Mix English words naturally when appropriate.",
            'example': "Namaste! Nenu ippudu chala bagunnanu. Meeru ela unnaru?"
        },
        
        'bengali': {
            'instruction': """Respond in Bengali language but write using English letters (Roman script). 
            Use Bengali words, phrases, and expressions but spell them in English letters.""",
            'style_note': "Use Bengali vocabulary in Roman letters.",
            'example': "Namaskar! Ami aaj bhalo achhi. Apni kemon achen?"
        },
        
        'gujarati': {
            'instruction': """Respond in Gujarati language but write using English letters (Roman script). 
            Use Gujarati words, phrases, and expressions but spell them in English letters.""",
            'style_note': "Use Gujarati vocabulary in Roman letters.",
            'example': "Namaste! Hu aaje saras chhu. Tame kem cho?"
        },
        
        'marathi': {
            'instruction': """Respond in Marathi language but write using English letters (Roman script). 
            Use Marathi words, phrases, and expressions but spell them in English letters.""",
            'style_note': "Use Marathi vocabulary in Roman letters.",
            'example': "Namaskar! Mi aaj khup bara aahe. Tumhi kase aahat?"
        },
        
        'punjabi': {
            'instruction': """Respond in Punjabi language but write using English letters (Roman script). 
            Use Punjabi words, phrases, and expressions but spell them in English letters.""",
            'style_note': "Use Punjabi vocabulary in Roman letters.",
            'example': "Sat Sri Akal! Main aaj bahut vadiya haan. Tussi kive ho?"
        },
        
        'kannada': {
            'instruction': """Respond in Kannada language but write using English letters (Roman script). 
            Use Kannada words, phrases, and expressions but spell them in English letters.""",
            'style_note': "Use Kannada vocabulary in Roman letters.",
            'example': "Namaskara! Naanu indu tumba chennagi iddini. Neevu heege iddira?"
        },
        
        'malayalam': {
            'instruction': """Respond in Malayalam language but write using English letters (Roman script). 
            Use Malayalam words, phrases, and expressions but spell them in English letters.""",
            'style_note': "Use Malayalam vocabulary in Roman letters.",
            'example': "Namaskaram! Njan innu valare nannayittund. Ningal engane und?"
        },
        
        'spanish': {
            'instruction': """Respond in Spanish but keep it simple and readable in Roman script.""",
            'style_note': "Use Spanish vocabulary but keep it accessible.",
            'example': "Hola! Estoy muy bien hoy. Como estas tu?"
        },
        
        'french': {
            'instruction': """Respond in French but keep it simple and readable in Roman script.""",
            'style_note': "Use French vocabulary but keep it accessible.",
            'example': "Bonjour! Je vais tres bien aujourd'hui. Comment allez-vous?"
        }
    }
    
    @classmethod
    def get_romanized_prompt(cls, target_language: str, character_name: str) -> str:
        """Generate romanized language prompt"""
        target_language = target_language.lower()
        
        if target_language not in cls.ROMANIZED_LANGUAGE_INSTRUCTIONS:
            target_language = 'english'
        
        lang_config = cls.ROMANIZED_LANGUAGE_INSTRUCTIONS[target_language]
        
        prompt = f"""
LANGUAGE STYLE INSTRUCTION for {character_name}:

{lang_config['instruction']}

Style Notes: {lang_config['style_note']}

Example response style: "{lang_config['example']}"

IMPORTANT RULES:
1. Use ONLY English letters (Roman script) - no other scripts
2. Your personality and character traits remain EXACTLY the same
3. Only the language content and expressions change
4. Keep the same emotion, tone, and character voice
5. The user may type in English, but you respond in {target_language} using Roman letters

You are still {character_name} with all your unique traits, just expressing yourself in {target_language} style.
"""
        
        return prompt.strip()

class MultilingualSupport:
    """Romanized multilingual support"""
    
    def __init__(self, db_session=None):
        self.db = db_session
        self.adapter = RomanizedLanguageAdapter()
    
    def create_romanized_prompt(self, base_prompt: str, character_name: str, 
                               target_language: str, rag_context: str, conversation_history: str) -> str:
        """Create prompt for romanized language switching"""
        
        # Get romanized language instructions
        language_prompt = self.adapter.get_romanized_prompt(target_language, character_name)
        
        complete_prompt = f"""{base_prompt}

{language_prompt}

Additional Knowledge:
{rag_context}

Conversation History:
{conversation_history}

{character_name}:"""
        
        return complete_prompt

# Updated language configuration for romanized display
SUPPORTED_LANGUAGES = {
    'english': {'name': 'English', 'native_name': 'English', 'code': 'en'},
    'hindi': {'name': 'Hindi', 'native_name': 'Hindi (Roman)', 'code': 'hi'},
    'tamil': {'name': 'Tamil', 'native_name': 'Tamil (Roman)', 'code': 'ta'},
    'telugu': {'name': 'Telugu', 'native_name': 'Telugu (Roman)', 'code': 'te'},
    'bengali': {'name': 'Bengali', 'native_name': 'Bengali (Roman)', 'code': 'bn'},
    'gujarati': {'name': 'Gujarati', 'native_name': 'Gujarati (Roman)', 'code': 'gu'},
    'marathi': {'name': 'Marathi', 'native_name': 'Marathi (Roman)', 'code': 'mr'},
    'punjabi': {'name': 'Punjabi', 'native_name': 'Punjabi (Roman)', 'code': 'pa'},
    'kannada': {'name': 'Kannada', 'native_name': 'Kannada (Roman)', 'code': 'kn'},
    'malayalam': {'name': 'Malayalam', 'native_name': 'Malayalam (Roman)', 'code': 'ml'},
    'spanish': {'name': 'Spanish', 'native_name': 'Español', 'code': 'es'},
    'french': {'name': 'French', 'native_name': 'Français', 'code': 'fr'},
}

def get_supported_languages_list() -> List[Dict]:
    """Get list of supported romanized languages"""
    return [
        {
            'code': code,
            'name': info['name'],
            'native_name': info['native_name'],
            'iso_code': info['code']
        }
        for code, info in SUPPORTED_LANGUAGES.items()
    ]

def detect_language(text: str) -> str:
    """Simple detection - mostly for analytics since all input will be Roman"""
    # Since everything is Roman script, just return English for analytics
    return 'english'