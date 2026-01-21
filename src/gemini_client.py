"""Gemini API client functions for AI-powered analysis."""

import os
import re
import json
import time
import streamlit as st

try:
    from google import genai
except Exception:
    genai = None


def gemini_generate(prompt: str, model_name: str = 'gemini-2.0-flash', retries: int = 2, backoff: float = 2.0) -> str | None:
    """
    Generate content using Gemini API.
    
    Args:
        prompt: The prompt to send to Gemini
        model_name: The Gemini model to use
        retries: Number of retries on rate limit errors
        backoff: Backoff multiplier for retries
        
    Returns:
        Generated text or None on failure
    """
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or genai is None:
        return None

    try:
        client = genai.Client(api_key = api_key)
        last_call = st.session_state.get("gemini_last_call", 0.0)
        now = time.time()
        min_interval = 1.2
        if now - last_call < min_interval:
            time.sleep(min_interval - (now - last_call))

        for attempt in range(retries + 1):
            try:
                response = client.models.generate_content(
                    model = model_name,
                    contents = prompt
                )
                st.session_state["gemini_last_call"] = time.time()
                return response.text.strip()
            except Exception as e:
                st.warning(f'Gemini attempt {attempt + 1} error: {e}')
                if "429" in str(e) and attempt < retries:
                    time.sleep(backoff * (attempt + 1))
                    continue
                raise
    except Exception as e:
        st.error(f'Gemini error: {type(e).__name__}: {e}')
        return None


def agent_truthfulness_assessment(text: str) -> dict | None:
    """
    Use Gemini to assess the truthfulness of a tweet.
    
    Args:
        text: The tweet text to analyze
        
    Returns:
        Dictionary with verdict, confidence, and bullets, or None on failure
    """
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or genai is None:
        return None

    try:
        prompt = (
            "You are an assistant assessing the plausibility of a tweet. "
            "Do NOT claim certainty. Respond strictly in JSON with keys: "
            "verdict (one of 'Likely True', 'Likely False', 'Uncertain'), "
            "confidence (0-100 integer), bullets (array of 2-4 short bullet points). "
            "Keep bullets concise and focus on clarity, evidence, specificity, and source cues.\n\n"
            f"Tweet:\n{text}\n"
        )
        response_text = gemini_generate(prompt)
        if not response_text:
            return None
        
        response_text = response_text.strip()
        
        if "```json" in response_text:
            match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if match:
                response_text = match.group(1).strip()
        elif "```" in response_text:
            match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
            if match:
                response_text = match.group(1).strip()
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        try:
            data = json.loads(response_text)
            verdict = str(data.get("verdict", "")).strip() or "Uncertain"
            confidence = int(data.get("confidence", 0))
            bullets = data.get("bullets", [])
            if isinstance(bullets, str):
                bullets = [bullets]
            bullets = [str(b).strip() for b in bullets if str(b).strip()]
            return {
                "verdict": verdict,
                "confidence": max(0, min(100, confidence)),
                "bullets": bullets,
            }
        except json.JSONDecodeError:
            return None
    except Exception as e:
        st.warning(f'Gemini analysis error: {e}')
        return None


def agent_technical_explanation(text: str, prediction: str, model_confidence: float, 
                                real_prob: float, fake_prob: float) -> str | None:
    """
    Use Gemini to generate a technical explanation of the model's prediction.
    
    Args:
        text: The tweet text
        prediction: Model's prediction (REAL/FAKE)
        model_confidence: Model's confidence percentage
        real_prob: Probability of REAL
        fake_prob: Probability of FAKE
        
    Returns:
        Technical explanation text or None on failure
    """
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or genai is None:
        return None

    try:
        prompt = (
            "You are explaining a model decision for a fake-news classifier. "
            "Be technical: reference probability scores, tokenization, and metadata features "
            "in general terms. Do not claim access to internal weights. Provide 3-6 bullets.\n\n"
            f"Model prediction: {prediction}\n"
            f"Confidence: {model_confidence:.1f}%\n"
            f"Probabilities - REAL: {real_prob:.1f}%, FAKE: {fake_prob:.1f}%\n"
            f"Tweet:\n{text}\n"
        )
        response_text = gemini_generate(prompt)
        return response_text.strip() if response_text else None
    except Exception as e:
        st.warning(f'Gemini technical explanation error: {e}')
        return None
