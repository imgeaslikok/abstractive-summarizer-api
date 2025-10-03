import logging
from typing import Any, Dict, Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class SummarizerService:
    """
    Manages the loading and access of the Transformer model pipeline.
    Uses the Singleton pattern to ensure the model is loaded only once per process.
    """

    _instance: Optional["SummarizerService"] = None
    _is_model_loaded: bool = False

    # Model parameters
    MODEL_NAME: str = "facebook/bart-large-cnn"

    # Model instance properties
    pipeline: Optional[pipeline] = None

    def __new__(cls, *args, **kwargs):
        """
        Ensures only one instance of the service exists.
        """
        if cls._instance is None:
            cls._instance = super(SummarizerService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Constructor is lightweight; actual model loading is lazy.
        """
        if not hasattr(self, "_initialized"):
            # This flag prevents re-initialization on subsequent Singleton calls
            self._initialized = True
            logger.info(f"SummarizerService initialized. Model: {self.MODEL_NAME}")

    def load_model(self):
        """
        Loads the model into memory. Designed to be called lazily on the first request.
        """
        if SummarizerService._is_model_loaded and self.pipeline is not None:
            logger.info("Model already loaded. Skipping initialization.")
            return

        logger.info(f"--- INFO: Model loading started: {self.MODEL_NAME} ---")
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.MODEL_NAME, local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME, local_files_only=True
            )

            self.pipeline = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=-1,
            )

            SummarizerService._is_model_loaded = True
            logger.info("+++ SUCCESS: Summarization model loaded successfully. +++")

        except Exception as e:
            SummarizerService._is_model_loaded = False
            logger.critical(f"!!! CRITICAL ERROR: Model failed to load: {e} !!!")

    def get_pipeline(self) -> Optional[pipeline]:
        """
        Returns the loaded NLP pipeline instance.
        """
        if not SummarizerService._is_model_loaded:
            self.load_model()

        return self.pipeline

    def get_status(self) -> Dict[str, Any]:
        """
        Returns the current status of the model.
        """
        return {
            "model_name": self.MODEL_NAME,
            "status": "Loaded" if SummarizerService._is_model_loaded else "Loading",
            "is_ready": SummarizerService._is_model_loaded,
        }


# Helper function for Dependency Injection
def get_summarizer_service() -> "SummarizerService":
    """
    Returns the Singleton instance of the SummarizerService.
    """
    return SummarizerService()
