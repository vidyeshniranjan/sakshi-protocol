def mauna_logic_gate(output_stream, confidence_score):
    """
    If confidence is low, trigger a hard-coded recalibration message 
    instead of letting the LLM continue to hallucinate.
    """
    SAFE_THRESHOLD = 0.85
    RECALIBRATION_MSG = "System is recalibrating for accuracy. Internal logic drift detected."

    if confidence_score >= SAFE_THRESHOLD:
        return output_stream
    else:
        # Instead of 'None', we return a safe, deterministic string.
        # This prevents the AI from 'guessing' a reason.
        return RECALIBRATION_MSG
