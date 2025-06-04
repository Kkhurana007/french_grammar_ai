# AI model management for grammar and syntax checking


from transformers import pipeline

class AIModels:
    def __init__(self):
        self.grammar_model = None
        self.syntax_model = None
    
    def load_models(self):
        self.grammar_model = pipeline("text2text-generation", model="grammarly/coedit-large")
        self.syntax_model = pipeline("token-classification", model="qanastek/pos-french")