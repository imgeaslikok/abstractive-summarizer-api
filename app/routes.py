import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas import ModelStatus, PredictionOutput, TextInput
from model.summarization import SummarizerService, get_summarizer_service

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1", tags=["summarization"])


@router.get("/status", response_model=ModelStatus)
def get_model_status(service: SummarizerService = Depends(get_summarizer_service)):
    """
    Readiness Check: Returns the current loading status of the AI model.
    The response changes from 'Loading' to 'Loaded' once the model is in memory.
    """
    logger.info("Model status check requested.")

    status_data = service.get_status()

    return ModelStatus(**status_data)


@router.post("/summarize", response_model=PredictionOutput)
def summarize_text(
    input_data: TextInput, service: SummarizerService = Depends(get_summarizer_service)
):
    """
    Abstractive summarization endpoint.
    Processes the input text using the loaded LLM pipeline.
    """
    pipeline_instance = service.get_pipeline()

    if pipeline_instance is None:
        logger.warning("Summarization request received, but model is not yet loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet ready. Please check the /status endpoint.",
        )

    # Prepare generation parameters from the input schema
    generation_kwargs: Dict[str, Any] = {
        "max_length": input_data.max_length,
        "min_length": input_data.min_length,
        "num_beams": input_data.num_beams,
        "repetition_penalty": input_data.repetition_penalty,
        "do_sample": False,  # Use deterministic beam search
    }

    try:
        logger.info(
            f"Generating summary with max_length={input_data.max_length} and num_beams={input_data.num_beams}"
        )

        summary_result = pipeline_instance(input_data.text, **generation_kwargs)

        # Extract the generated text
        summary = summary_result[0].get("summary_text", "Generation failed.")

        logger.info("Summary generation successful.")
        return PredictionOutput(summary=summary)

    except Exception as e:
        logger.error(f"Inference error during summarization: {e}")
        # Return 500 status code for internal model inference errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing text for summarization: {e}",
        )
