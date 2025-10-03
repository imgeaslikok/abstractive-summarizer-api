from pydantic import BaseModel, ConfigDict, Field

# --- Request/Response Schemas ---


class TextInput(BaseModel):
    """
    Schema for the input text and summarization parameters.
    """

    text: str = Field(
        ...,
        min_length=150,
        description="The document text to be summarized. Minimum 150 characters enforced.",
    )
    min_length: int = Field(
        50, description="Minimum number of tokens for the output summary."
    )
    max_length: int = Field(
        150, description="Maximum number of tokens for the output summary."
    )
    num_beams: int = Field(
        4,
        description="Number of beams for beam search, controlling the quality vs. speed tradeoff.",
    )
    repetition_penalty: float = Field(
        2.5, description="Penalty for token repetition during generation."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": (
                    "The latest quarterly report shows significant growth in the AI sector"
                    "driven by new large language model deployments."
                    "This success required robust architecture and high-performance APIs..."
                ),
                "min_length": 50,
                "max_length": 150,
                "num_beams": 4,
                "repetition_penalty": 2.5,
            }
        }
    )


class PredictionOutput(BaseModel):
    """
    Schema for the final summary output.
    """

    summary: str


# --- Monitoring Schemas ---


class StatusOutput(BaseModel):
    """
    Status check schema for monitoring (Liveness and Readiness).
    """

    status: str
    model_status: str


class ModelStatus(BaseModel):
    """
    Schema for the detailed model readiness status.
    """

    model_name: str
    status: str
    is_ready: bool
