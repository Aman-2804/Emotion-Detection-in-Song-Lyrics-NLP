"""
LLM evaluation with in-context learning for emotion detection.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import torch


class LLMEvaluator:
    """
    Evaluates LLMs using zero-shot and few-shot prompting for emotion classification.
    """
    
    def __init__(
        self,
        model_name: str,
        emotion_categories: List[str],
        device: Optional[str] = None
    ):
        """
        Initialize LLM evaluator.
        
        Args:
            model_name: HuggingFace model name (e.g., 'meta-llama/Llama-2-7b-hf')
            emotion_categories: List of emotion labels
            device: Device to run model on
        """
        self.model_name = model_name
        self.emotion_categories = emotion_categories
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )
        self.model.eval()
    
    def create_prompt(
        self,
        lyric: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        zero_shot: bool = True
    ) -> str:
        """
        Create prompt for emotion classification.
        
        Args:
            lyric: Input song lyric
            few_shot_examples: List of example dicts with 'lyric' and 'emotion'
            zero_shot: Whether to use zero-shot (no examples)
            
        Returns:
            Formatted prompt string
        """
        emotion_str = ', '.join(self.emotion_categories)
        
        prompt = f"Task: Classify the emotion in the following song lyric.\n\n"
        prompt += f"Emotion categories: {emotion_str}\n\n"
        
        if not zero_shot and few_shot_examples:
            prompt += "Examples:\n\n"
            for example in few_shot_examples:
                prompt += f'Lyric: "{example["lyric"]}" → Emotion: {example["emotion"]}\n\n'
        
        prompt += f'Lyric: "{lyric}"\n\n'
        prompt += "Emotion:"
        
        return prompt
    
    def predict(
        self,
        lyric: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        zero_shot: bool = True,
        max_new_tokens: int = 10
    ) -> str:
        """
        Predict emotion for a lyric using LLM.
        
        Args:
            lyric: Input song lyric
            few_shot_examples: Optional few-shot examples
            zero_shot: Whether to use zero-shot
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Predicted emotion label
        """
        prompt = self.create_prompt(lyric, few_shot_examples, zero_shot)
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract emotion from generated text
        emotion = self._extract_emotion(generated_text)
        
        return emotion
    
    def _extract_emotion(self, generated_text: str) -> str:
        """
        Extract emotion label from generated text.
        
        Args:
            generated_text: Full generated text including prompt
            
        Returns:
            Extracted emotion label
        """
        # Try to find emotion after "Emotion:" marker
        if "Emotion:" in generated_text:
            emotion_part = generated_text.split("Emotion:")[-1].strip()
            # Take first word/token
            emotion = emotion_part.split()[0].strip().lower()
            
            # Map to valid category
            for cat in self.emotion_categories:
                if cat.lower() == emotion or cat.lower() in emotion:
                    return cat
        
        # Fallback: return first valid category
        return self.emotion_categories[0]
