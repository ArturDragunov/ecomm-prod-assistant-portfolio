from enum import Enum
from typing import Dict
import string


class PromptType(str, Enum):
    PRODUCT_BOT = "product_bot"
    # REVIEW_BOT = "review_bot"
    # COMPARISON_BOT = "comparison_bot"

# wrapper around Python's string formatting with validation and versioning
class PromptTemplate:
    def __init__(self, template: str, description: str = "", version: str = "v1"):
        self.template = template.strip() # strip whitespaces from the template. Example of the whitespace is \n\n
        self.description = description
        self.version = version

    def format(self, **kwargs) -> str:
        """Instead of just calling template.format(**kwargs) and potentially getting
        confusing errors, it:
        Pre-validates that all required placeholders are provided
        Gives clear error messages if something is missing
        Prevents runtime errors from missing variables
        
        # Template with placeholders
        template = "Hello {name}, you have {count} messages from {sender}"

        # Your class automatically detects: ['name', 'count', 'sender']

        # Safe formatting with validation
        prompt.format(name="Alice", count=5, sender="Bob")  # ✅ Works
        prompt.format(name="Alice")  # ❌ Raises: "Missing placeholders: ['count', 'sender']"    

        Without this class:
        template = "Hello {name}, you have {count} messages"
        template.format(name="Alice")  # ❌ KeyError: 'count' - confusing!

        With class:
        prompt.format(name="Alice")  # ❌ ValueError: "Missing placeholders: ['count']" - clear!
        """
        # Validate placeholders before formatting
        missing = [
            f for f in self.required_placeholders() if f not in kwargs
        ]
        if missing:
            raise ValueError(f"Missing placeholders: {missing}")
        return self.template.format(**kwargs)

    def required_placeholders(self):
        """This uses Python's string.Formatter().parse() to automatically
         detect all {placeholder} variables in the template."""

        return [field_name for _, field_name, _, _ in string.Formatter().parse(self.template) if field_name]


# Central Registry -> our main prompt
PROMPT_REGISTRY: Dict[PromptType, PromptTemplate] = { # prompt registry is a dictionary that maps a prompt type to a prompt template
    PromptType.PRODUCT_BOT: PromptTemplate(
        """
        You are an expert EcommerceBot specialized in product recommendations and handling customer queries.
        Analyze the provided product titles, ratings, and reviews to provide accurate, helpful responses.
        Stay relevant to the context, and keep your answers concise and informative.

        CONTEXT:
        {context}

        QUESTION: {question}

        YOUR ANSWER:
        """,
        description="Handles ecommerce QnA & product recommendation flows"
    )
}
