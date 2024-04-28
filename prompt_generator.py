def generate_universal_prompt(question):
    """
    Generates a prompt for the Gemini API to provide a detailed explanation and request Manim visualization.
    This function creates a more structured and detailed prompt to handle complex queries.
    """
    return f"Please provide a comprehensive explanation for {question}:"