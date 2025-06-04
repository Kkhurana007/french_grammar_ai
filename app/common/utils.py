# utility functions for the application

"""
Common utilities for French Grammar AI
Includes POS tagging, sentence analysis, and helper functions
"""

from typing import List, Dict, Tuple, Optional
import spacy
from spacy.tokens import Token, Doc
import re
from collections import defaultdict

# French POS tags mapping (imported from constants.py)
from .constants import FRENCH_POS_TAGS

# Load French language model
nlp = spacy.load("fr_core_news_lg")

def get_french_pos_tag(spacy_tag: str) -> str:
    """
    Convert spaCy POS tag to French grammatical term
    Args:
        spacy_tag (str): spaCy POS tag (e.g., 'NOUN', 'VERB')
    Returns:
        str: French grammatical term (e.g., 'Nom', 'Verbe')
    """
    return FRENCH_POS_TAGS.get(spacy_tag, spacy_tag)

def analyze_sentence_structure(sentence: str) -> List[Dict]:
    """
    Analyze a French sentence and return its grammatical structure
    Args:
        sentence (str): Input French sentence
    Returns:
        List[Dict]: List of tokens with French grammatical analysis
    """
    doc = nlp(sentence)
    structure = []
    
    for token in doc:
        token_info = {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "french_pos": get_french_pos_tag(token.pos_),
            "tag": token.tag_,
            "dep": token.dep_,
            "is_punctuation": token.is_punct,
            "is_stop": token.is_stop,
        }
        
        # Special handling for subject detection
        if token.dep_ in ("nsubj", "nsubj:pass"):
            token_info["french_pos"] = "Sujet"
        
        structure.append(token_info)
    
    return structure

def extract_sentence_components(doc: Doc) -> Dict[str, List]:
    """
    Extract key grammatical components from a parsed sentence
    Args:
        doc (spacy.Doc): Parsed sentence
    Returns:
        Dict[str, List]: Dictionary of components by grammatical category
    """
    components = defaultdict(list)
    
    for token in doc:
        category = get_french_pos_tag(token.pos_)
        
        # Special cases
        if token.dep_ in ("nsubj", "nsubj:pass"):
            category = "Sujet"
        elif token.pos_ == "AUX" and "VerbForm=Fin" in token.tag_:
            category = "Auxiliaire"
        
        components[category].append({
            "text": token.text,
            "lemma": token.lemma_,
            "tag": token.tag_,
            "dep": token.dep_
        })
    
    return dict(components)

def detect_compound_tenses(verb_phrase: List[Token]) -> Optional[str]:
    """
    Detect French compound tenses (passé composé, plus-que-parfait, etc.)
    Args:
        verb_phrase (List[Token]): List of tokens in a verb phrase
    Returns:
        Optional[str]: Tense name if detected, None otherwise
    """
    auxiliaries = [t for t in verb_phrase if t.pos_ == "AUX"]
    verbs = [t for t in verb_phrase if t.pos_ == "VERB" and t.lemma_ != "être" and t.lemma_ != "avoir"]
    
    if not auxiliaries or not verbs:
        return None
    
    # Check for passé composé
    if (len(auxiliaries) == 1 and auxiliaries[0].lemma_ in ("avoir", "être"):
        return "passé composé"
    
    # Check for plus-que-parfait
    if (len(auxiliaries) == 2 and 
        auxiliaries[0].lemma_ in ("avoir", "être") and 
        auxiliaries[1].lemma_ == "avoir"):
        return "plus-que-parfait"
    
    return None

def validate_adverb_placement(sentence: str) -> Tuple[bool, Optional[str]]:
    """
    Validate adverb placement in French sentences
    Args:
        sentence (str): Input sentence
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    doc = nlp(sentence)
    
    for token in doc:
        if token.pos_ == "ADV":
            # Check if adverb is incorrectly placed between subject and verb
            if (token.head.pos_ == "VERB" and 
                any(t.dep_ in ("nsubj", "nsubj:pass") for t in token.head.lefts)):
                return (False, 
                       f"Adverbe '{token.text}' mal placé. En français, l'adverbe ne doit généralement pas séparer le sujet et le verbe.")
            
            # Check adverb placement in compound tenses
            if token.head.pos_ == "AUX" and token.i > token.head.i:
                tense = detect_compound_tenses([token.head] + [t for t in token.head.rights if t.pos_ == "VERB"])
                if tense:
                    return (False,
                           f"Dans le {tense}, l'adverbe '{token.text}' devrait normalement se placer après l'auxiliaire '{token.head.text}'.")
    
    return (True, None)

def highlight_pos_in_sentence(sentence: str, pos_tags: List[str]) -> str:
    """
    Highlight specific parts of speech in a sentence
    Args:
        sentence (str): Input sentence
        pos_tags (List[str]): POS tags to highlight (e.g., ['VERB', 'ADV'])
    Returns:
        str: Sentence with highlighted components
    """
    doc = nlp(sentence)
    highlighted = []
    
    for token in doc:
        if token.pos_ in pos_tags:
            highlighted.append(f"[{token.text}]({get_french_pos_tag(token.pos_)})")
        else:
            highlighted.append(token.text)
    
    return " ".join(highlighted)

def is_french_sentence(text: str) -> bool:
    """
    Basic check if text appears to be French
    Args:
        text (str): Input text
    Returns:
        bool: True if likely French
    """
    # Check for common French words
    french_indicators = [
        r'\b(le|la|les|un|une|des)\b',
        r'\b(je|tu|il|elle|nous|vous|ils|elles)\b',
        r'\b(et|mais|ou|donc|car|ni|or)\b',
        r'\b(à|de|en|pour|dans|sur|avec)\b'
    ]
    
    return any(re.search(pattern, text.lower()) for pattern in french_indicators)

def lemmatize_french_text(text: str) -> str:
    """
    Lemmatize French text (reduce words to base form)
    Args:
        text (str): Input text
    Returns:
        str: Lemmatized text
    """
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_punct])