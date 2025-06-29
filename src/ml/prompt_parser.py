"""
GPT-4 based prompt parser for extracting ML task specifications
"""

import json
from typing import Dict, Any, Optional

import openai
from pydantic import BaseModel, ValidationError
import structlog

from ..core.config import settings
from ..models.schema import PromptRequest, PromptResponse

logger = structlog.get_logger()


class MLTaskSpec(BaseModel):
    """Pydantic model for ML task specification validation"""
    task: str  # "classification" or "regression"
    target: str
    metric: str  # recall, precision, f1, accuracy, rmse, mae
    constraints: Dict[str, Any] = {}


class PromptParser:
    """GPT-4 based prompt parser"""
    
    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        
        self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.system_prompt = self._get_system_prompt()
        self.few_shot_examples = self._get_few_shot_examples()
    
    async def parse_prompt(self, request: PromptRequest) -> PromptResponse:
        """Parse user prompt and extract ML task specification"""
        
        logger.info("Parsing prompt", session_id=request.session_id)
        
        # Prepare context with dataset preview
        context = self._prepare_context(request)
        
        try:
            # Call GPT-4
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    *self.few_shot_examples,
                    {"role": "user", "content": context}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            logger.info("GPT-4 response received", response_length=len(response_text))
            
            # Extract JSON from response
            task_spec = self._extract_json(response_text)
            
            # Validate with Pydantic
            validated_spec = MLTaskSpec(**task_spec)
            
            # Check for clarifications needed
            clarifications = self._check_clarifications_needed(
                validated_spec, request.dataset_preview
            )
            
            return PromptResponse(
                task=validated_spec.task,
                target=validated_spec.target,
                metric=validated_spec.metric,
                constraints=validated_spec.constraints,
                clarifications_needed=clarifications
            )
            
        except ValidationError as e:
            logger.error("Validation error in parsed response", error=str(e))
            raise ValueError(f"Failed to parse valid ML task specification: {str(e)}")
        
        except Exception as e:
            logger.error("Failed to parse prompt", error=str(e))
            raise ValueError(f"Prompt parsing failed: {str(e)}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for GPT-4"""
        return """You are a strict JSON generator for ML task specification extraction.

Given a user request and dataset preview, output a JSON object that adheres to this schema:

{
  "task": "classification" | "regression",
  "target": "column_name",
  "metric": "recall" | "precision" | "f1" | "accuracy" | "rmse" | "mae",
  "constraints": {
    "feature_limit": int | null,
    "exclude_cols": [string] (optional),
    "class_weight": "balanced" | null (optional)
  }
}

RULES:
1. ONLY output valid JSON matching this schema
2. target must be an existing column name from the dataset
3. metric must match the task type (classification: recall/precision/f1/accuracy, regression: rmse/mae)
4. If request is ambiguous, choose the most reasonable interpretation
5. exclude_cols should include obviously non-predictive columns (IDs, names, etc.)
6. feature_limit should be set if user specifies a max number of features

NO additional text or explanation - JSON only."""

    def _get_few_shot_examples(self) -> list:
        """Get few-shot examples for prompt engineering"""
        return [
            {
                "role": "user",
                "content": """Dataset: customers.csv (24 columns including CustomerID, Churn, MonthlyCharges, TotalCharges)
User request: Predict churn. Optimize recall. Max 5 features. Exclude CustomerID."""
            },
            {
                "role": "assistant", 
                "content": json.dumps({
                    "task": "classification",
                    "target": "Churn", 
                    "metric": "recall",
                    "constraints": {
                        "feature_limit": 5,
                        "exclude_cols": ["CustomerID"],
                        "class_weight": "balanced"
                    }
                })
            },
            {
                "role": "user",
                "content": """Dataset: housing.csv (columns: price, bedrooms, bathrooms, sqft, location, property_id)
User request: Predict house prices using all features except property_id. Focus on RMSE."""
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "task": "regression",
                    "target": "price",
                    "metric": "rmse", 
                    "constraints": {
                        "feature_limit": None,
                        "exclude_cols": ["property_id"]
                    }
                })
            }
        ]
    
    def _prepare_context(self, request: PromptRequest) -> str:
        """Prepare context for GPT-4 including dataset info"""
        
        preview = request.dataset_preview or {}
        columns = preview.get("columns", [])
        shape = preview.get("shape", [0, 0])
        
        context = f"""Dataset: {shape[0]} rows, {shape[1]} columns
Columns: {", ".join(columns[:20])}{"..." if len(columns) > 20 else ""}

User request: {request.prompt}"""
        
        return context
    
    def _extract_json(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from GPT-4 response"""
        
        # Try to find JSON in response
        try:
            # Look for JSON block
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[start:end]
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON", response=response_text, error=str(e))
            raise ValueError(f"Invalid JSON in response: {str(e)}")
    
    def _check_clarifications_needed(
        self, 
        spec: MLTaskSpec, 
        dataset_preview: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Check if clarifications are needed"""
        
        if not dataset_preview:
            return None
        
        columns = dataset_preview.get("columns", [])
        
        # Check if target column exists
        if spec.target not in columns:
            available = ", ".join(columns[:10])
            return f"Target column '{spec.target}' not found. Available columns: {available}..."
        
        # Check for potential issues
        issues = []
        
        # Check for high null columns that should be excluded
        if "sample_rows" in dataset_preview:
            sample_df = dataset_preview["sample_rows"]
            for col in columns:
                if col not in spec.constraints.get("exclude_cols", []):
                    # Simple null check on sample
                    null_count = sum(1 for row in sample_df if not row.get(col))
                    if null_count > len(sample_df) * 0.8:
                        issues.append(f"Column '{col}' has high null values")
        
        if issues:
            return "Data quality issues detected: " + "; ".join(issues) + ". Should these columns be excluded?"
        
        return None 