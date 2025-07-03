"""
GPT-4 based prompt parser for extracting ML task specifications
"""

import json
import re
from typing import Dict, Any, Optional, List, Tuple

import openai
from pydantic import BaseModel, ValidationError
import structlog
import pandas as pd

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
            
            # ENHANCED: Add fuzzy column matching
            if request.dataset_preview and request.dataset_preview.get("columns"):
                validated_spec.target = self._find_best_column_match(
                    validated_spec.target, 
                    request.dataset_preview["columns"]
                )
            
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
            
            # Try fallback parsing first
            fallback_result = self._fallback_parse(request)
            if fallback_result:
                return fallback_result
            
            raise ValueError(f"Prompt parsing failed: {str(e)}")
    
    def _fallback_parse(self, prompt_request: PromptRequest) -> Optional[PromptResponse]:
        """Enhanced fallback parsing with intelligent heuristics"""
        try:
            prompt = prompt_request.prompt.lower()
            dataset_preview = prompt_request.dataset_preview
            
            if not dataset_preview:
                return None
            
            columns = dataset_preview.get("columns", [])
            sample_rows = dataset_preview.get("sample_rows", [])
            
            # Enhanced clustering detection
            clustering_keywords = [
                "cluster", "segment", "group", "pattern", "unsupervised", 
                "customer segments", "behavior patterns", "market segments",
                "find groups", "categorize customers", "group similar",
                "identify patterns", "discover segments", "analyze behavior"
            ]
            clustering_phrases = [
                "segment", "group", "categorize", "cluster", "behavior pattern",
                "customer group", "market segment", "find similar", "identify group"
            ]
            
            is_clustering = any(keyword in prompt for keyword in clustering_keywords)
            if not is_clustering:
                # Check for clustering phrases
                is_clustering = any(phrase in prompt for phrase in clustering_phrases)
            
            if is_clustering:
                # For clustering, check if no specific target is mentioned
                target_mentioned = self._find_target_column_from_prompt(prompt, columns)
                if not target_mentioned:
                    return PromptResponse(
                        task="clustering",
                        target="",
                        metric="silhouette",
                        constraints=self._extract_constraints_from_prompt(prompt, columns),
                        clarifications_needed=None
                    )
            
            # Enhanced target column detection
            target_col = self._find_target_column_from_prompt(prompt, columns)
            
            # If no explicit target found, use intelligent fallback
            if not target_col:
                target_col = self._find_likely_target_column(columns, sample_rows, prompt)
            
            if not target_col:
                return None  # Can't determine target
            
            # Enhanced task type detection
            task_type, metric = self._determine_task_type(prompt, target_col, sample_rows)
            
            # Extract constraints from prompt
            constraints = self._extract_constraints_from_prompt(prompt, columns)
            
            return PromptResponse(
                task=task_type,
                target=target_col,
                metric=metric,
                constraints=constraints,
                clarifications_needed=None
            )
            
        except Exception as e:
            logger.error("Enhanced fallback parsing failed", error=str(e))
            return None
    
    def _find_target_column_from_prompt(self, prompt: str, columns: List[str]) -> Optional[str]:
        """Enhanced target column detection from prompt"""
        
        # Direct prediction patterns
        predict_patterns = [
            r"predict\s+(\w+)", r"predicting\s+(\w+)", r"forecast\s+(\w+)",
            r"estimate\s+(\w+)", r"target\s+(?:is\s+)?(\w+)", r"outcome\s+(?:is\s+)?(\w+)"
        ]
        
        for pattern in predict_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            for match in matches:
                # Find best column match
                best_match = self._find_best_column_match(match, columns)
                if best_match:
                    return best_match
        
        # Business outcome patterns
        business_patterns = {
            "churn": ["churn", "churned", "left", "attrition"],
            "risk": ["risk", "default", "failure", "bad"],
            "price": ["price", "cost", "value", "amount", "revenue", "sales"],
            "rating": ["rating", "score", "satisfaction", "quality"],
            "fraud": ["fraud", "fraudulent", "suspicious"],
            "conversion": ["convert", "conversion", "purchase", "buy"]
        }
        
        for intent, keywords in business_patterns.items():
            if any(keyword in prompt for keyword in keywords):
                # Look for columns that match this intent
                for col in columns:
                    if any(keyword in col.lower() for keyword in keywords):
                        return col
        
        # Column names mentioned in prompt
        for col in columns:
            col_variations = [
                col.lower(),
                col.lower().replace('_', ' '),
                col.lower().replace(' ', '_')
            ]
            
            for variation in col_variations:
                if variation in prompt and len(variation) > 2:  # Avoid short matches
                    return col
        
        return None
    
    def _find_likely_target_column(self, columns: List[str], sample_rows: List[Dict], prompt: str) -> Optional[str]:
        """Find most likely target column using heuristics"""
        
        candidates = []
        
        # Target indicators in column names
        target_indicators = [
            "target", "label", "class", "outcome", "result", "y", "dependent",
            "churn", "price", "amount", "value", "score", "rating", "risk",
            "default", "fraud", "conversion", "purchase", "success", "failure"
        ]
        
        for col in columns:
            col_lower = col.lower()
            
            # Skip obvious non-targets
            if any(skip in col_lower for skip in ["id", "key", "name", "date", "time", "created", "updated"]):
                continue
            
            score = 0
            
            # High score for explicit target indicators
            if any(indicator in col_lower for indicator in target_indicators):
                score += 10
            
            # Medium score for last column (common convention)
            if col == columns[-1]:
                score += 5
            
            # Check data characteristics if sample available
            if sample_rows:
                col_values = [row.get(col) for row in sample_rows if row.get(col) is not None]
                if col_values:
                    unique_count = len(set(col_values))
                    
                    # Moderate number of unique values suggests target
                    if 2 <= unique_count <= min(20, len(col_values) * 0.8):
                        score += 3
                    
                    # Binary targets are often important
                    if unique_count == 2:
                        score += 2
            
            if score > 0:
                candidates.append((col, score))
        
        # Return highest scoring candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        # Final fallback: last column
        return columns[-1] if columns else None
    
    def _determine_task_type(self, prompt: str, target_col: str, sample_rows: List[Dict]) -> Tuple[str, str]:
        """Determines the ML task type (classification or regression) and appropriate metric based on:
        
        1. Explicit keywords in the user's prompt (e.g. "classify", "predict value")
        2. Target column name indicators (e.g. "price" suggests regression, "status" suggests classification)
        3. Analysis of sample target values (e.g. binary values suggest classification)
        
        Args:
            prompt: The user's natural language request
            target_col: Name of the target column to predict
            sample_rows: List of sample data rows for analyzing target values
            
        Returns:
            Tuple[str, str]: (task_type, metric) where:
                - task_type is either "classification" or "regression"
                - metric is an appropriate metric for that task (e.g. "accuracy", "rmse")
        """
        # Explicit task indicators
        classification_keywords = [
            "classify", "classification", "predict category", "predict class",
            "risk assessment", "fraud detection", "churn prediction",
            "binary", "categorical", "yes/no", "true/false"
        ]
        
        regression_keywords = [
            "regress", "regression", "predict value", "predict amount", 
            "predict price", "forecast", "estimate", "continuous",
            "numeric prediction", "sales forecast", "revenue prediction"
        ]
        
        # Check explicit keywords
        if any(keyword in prompt for keyword in classification_keywords):
            metric = self._choose_classification_metric(prompt)
            return "classification", metric
        
        if any(keyword in prompt for keyword in regression_keywords):
            metric = self._choose_regression_metric(prompt)
            return "regression", metric
        
        # Infer from target column name
        classification_indicators = [
            "class", "category", "type", "churn", "status", "risk", "fraud",
            "default", "success", "failure", "convert", "buy", "click"
        ]
        
        regression_indicators = [
            "price", "cost", "amount", "value", "revenue", "sales", "income",
            "salary", "age", "count", "quantity", "volume", "rate", "score"
        ]
        
        target_lower = target_col.lower()
        
        if any(indicator in target_lower for indicator in classification_indicators):
            return "classification", "accuracy"
        
        if any(indicator in target_lower for indicator in regression_indicators):
            return "regression", "rmse"
        
        # Analyze sample data if available
        if sample_rows and target_col in sample_rows[0]:
            sample_values = [row.get(target_col) for row in sample_rows if row.get(target_col) is not None]
            
            if sample_values:
                task, metric = self._analyze_target_values(sample_values, prompt)
                return task, metric
        
        # Default fallback
        return "classification", "accuracy"
    
    def _analyze_target_values(self, sample_values: List, prompt: str) -> Tuple[str, str]:
        """Analyze sample target values to determine task type"""
        
        unique_values = set(sample_values)
        is_all_numeric = all(isinstance(val, (int, float)) for val in sample_values)
        
        # Handle boolean/binary values
        if len(unique_values) == 2:
            if set(str(v).lower() for v in unique_values) & {"true", "false", "yes", "no", "0", "1"}:
                metric = self._choose_classification_metric(prompt)
                return "classification", metric
        
        # Small number of unique values suggests classification
        if len(unique_values) <= 10:
            metric = self._choose_classification_metric(prompt)
            return "classification", metric
        
        # For numeric values, check if they're discrete or continuous
        if is_all_numeric:
            # Check if all are integers
            all_integers = all(val == int(val) for val in sample_values if isinstance(val, (int, float)))
            
            if all_integers:
                min_val = min(sample_values)
                max_val = max(sample_values)
                value_range = max_val - min_val if len(unique_values) > 1 else 0
                
                # Small range of integers likely classification
                if value_range <= 20 and len(unique_values) <= 15:
                    return "classification", "accuracy"
                else:
                    return "regression", "rmse"
            else:
                # Float values suggest regression
                return "regression", "rmse"
        
        # Non-numeric values suggest classification
        return "classification", "accuracy"
    
    def _choose_classification_metric(self, prompt: str) -> str:
        """Choose appropriate classification metric based on prompt context"""
        
        if any(word in prompt for word in ["recall", "sensitivity", "catch", "detect"]):
            return "recall"
        elif any(word in prompt for word in ["precision", "exact", "avoid false"]):
            return "precision"
        elif any(word in prompt for word in ["f1", "balance", "harmonic"]):
            return "f1"
        else:
            return "accuracy"
    
    def _choose_regression_metric(self, prompt: str) -> str:
        """Choose appropriate regression metric based on prompt context"""
        
        if any(word in prompt for word in ["mae", "absolute", "outlier"]):
            return "mae"
        else:
            return "rmse"
    def _extract_constraints_from_prompt(self, prompt: str, columns: List[str]) -> Dict[str, Any]:
        """Extract constraints from prompt text
        
        This method analyzes the user's prompt text to extract various constraints and preferences
        for the ML task. It looks for:

        1. Feature limits - How many features to use (e.g. "use top 10 features")
        2. Time budget - Whether speed or accuracy is prioritized 
        3. Columns to exclude - Both explicit exclusions and automatic ID/timestamp columns
        4. Class weighting - Whether to use balanced class weights for imbalanced data

        Args:
            prompt: The user's prompt text to analyze
            columns: List of available column names in the dataset

        Returns:
            Dict containing extracted constraints like:
            {
                "feature_limit": int,  # Max number of features to use
                "time_budget": "fast"|"medium"|"slow",  # Speed vs accuracy preference
                "exclude_cols": [str],  # Columns to exclude
                "class_weight": "balanced"|None  # Whether to use balanced weights
            }
        """
        
        constraints = {}
        
        # Extract feature limit using regex patterns
        # Matches phrases like "max 10 features", "use 5 features", etc.
        feature_patterns = [
            r"(?:max|maximum|top|best)\s+(\d+)\s+feature",
            r"(\d+)\s+features?\s+(?:max|maximum)", 
            r"limit.*?(\d+)\s+features?",
            r"use.*?(\d+)\s+features?"
        ]
        
        for pattern in feature_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            if matches:
                try:
                    constraints["feature_limit"] = int(matches[0])
                    break
                except ValueError:
                    continue
        
        # Determine time budget based on speed/accuracy keywords
        if any(word in prompt for word in ["quick", "fast", "rapid", "speed"]):
            constraints["time_budget"] = "fast"
        elif any(word in prompt for word in ["thorough", "comprehensive", "detailed", "slow", "accuracy"]):
            constraints["time_budget"] = "slow"
        else:
            constraints["time_budget"] = "medium"
        
        # Build list of columns to exclude
        exclude_cols = []
        
        # Look for explicit exclusions in prompt
        exclude_patterns = [
            r"exclude\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"without\s+([a-zA-Z_][a-zA-Z0-9_]*)", 
            r"ignore\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        ]
        
        for pattern in exclude_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            for match in matches:
                # Find closest matching column name
                best_match = self._find_best_column_match(match, columns)
                if best_match:
                    exclude_cols.append(best_match)
        
        # Auto-exclude common non-predictive columns
        for col in columns:
            col_lower = col.lower()
            if any(skip_word in col_lower for skip_word in ["id", "_id", "key", "index", "timestamp", "created", "updated"]):
                exclude_cols.append(col)
        
        if exclude_cols:
            constraints["exclude_cols"] = list(set(exclude_cols))  # Remove duplicates
        
        # Check for class balancing keywords
        if any(word in prompt for word in ["balanced", "imbalanced", "unbalanced", "rare", "minority"]):
            constraints["class_weight"] = "balanced"
        
        return constraints
    
    def _get_system_prompt(self) -> str:
        """Get the enhanced system prompt for GPT-4"""
        return """You are an expert ML consultant specializing in translating business questions into ML task specifications.

Your role is to understand user intent - even from conversational, ambiguous, or incomplete prompts - and extract a precise ML task specification.

Given a user request and dataset preview, output ONLY a JSON object that adheres to this schema:

{
  "task": "classification" | "regression" | "clustering",
  "target": "column_name" | "" (empty for clustering),
  "metric": "recall" | "precision" | "f1" | "accuracy" | "rmse" | "mae" | "silhouette",
  "constraints": {
    "feature_limit": int | null,
    "exclude_cols": [string] (optional),
    "class_weight": "balanced" | null (optional),
    "time_budget": "fast" | "medium" | "slow" (optional)
  }
}

ENHANCED INTERPRETATION RULES:
1. Be flexible and interpretive - understand business intent behind casual language
2. For ambiguous prompts, choose the most reasonable ML interpretation
3. Infer task type from context: business outcomes usually = classification, numerical predictions = regression
4. Smart column matching: "predict price" → look for "price", "house_price", "total_price", etc.
5. Automatically exclude obvious non-predictive columns (IDs, names, timestamps unless relevant)
6. Set balanced class weights for classification when target seems imbalanced
7. Infer time constraints from language like "quick", "fast", "thorough", "comprehensive"

BUSINESS LANGUAGE UNDERSTANDING:
- "analyze patterns" → clustering (no target)
- "predict whether..." → classification  
- "estimate/forecast how much..." → regression
- "find groups/segments" → clustering
- "risk assessment" → classification (risk as target)
- "pricing model" → regression (price as target)
- "customer behavior" → could be clustering or classification based on context

SMART DEFAULTS:
- Classification metrics: accuracy (general), recall (risk/medical), precision (spam/fraud)
- Regression metrics: rmse (general), mae (when outliers matter)
- Auto-exclude: any column with "id", "name", "timestamp", "date" (unless explicitly mentioned)
- Feature limits: suggest reasonable limits for high-dimensional data (>50 features)

OUTPUT: JSON only, no explanations or additional text."""
    def _get_few_shot_examples(self) -> list:
        """Get comprehensive few-shot examples for various scenarios
        
        This method provides example conversations between users and the assistant to help GPT-4 understand
        how to parse different types of ML task requests. Each example shows:
        
        1. A user request - ranging from technical to casual business language
        2. The expected JSON response with task specifications
        
        The examples cover common scenarios like:
        - Churn prediction (classification)
        - Sales forecasting (regression) 
        - Employee attrition (classification)
        - Customer segmentation (clustering)
        - Risk assessment (classification)
        - Property valuation (regression)
        - Scientific/medical predictions
        
        Each example demonstrates:
        - Task type inference from context
        - Target column identification
        - Appropriate metric selection
        - Smart constraint handling (feature limits, excluded columns)
        - Class weight balancing when needed
        - Time budget interpretation
        
        Returns:
            list: List of dictionaries containing role:user/assistant pairs with example conversations
        """
        return [
            # Clear, specific technical prompt
            {
                "role": "user",
                "content": """Dataset: customers.csv (24 columns including CustomerID, Churn, MonthlyCharges, TotalCharges, Contract, PaymentMethod)
User request: Predict churn. Optimize recall. Max 10 features. Exclude CustomerID."""
            },
            {
                "role": "assistant", 
                "content": json.dumps({
                    "task": "classification",
                    "target": "Churn", 
                    "metric": "recall",
                    "constraints": {
                        "feature_limit": 10,
                        "exclude_cols": ["CustomerID"],
                        "class_weight": "balanced"
                    }
                })
            },
            
            # Ambiguous business language
            {
                "role": "user",
                "content": """Dataset: sales_data.csv (18 columns including product_id, customer_age, purchase_amount, region, season, marketing_spend)
User request: I want to understand what drives our sales numbers and build a model to forecast revenue."""
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "task": "regression",
                    "target": "purchase_amount",
                    "metric": "rmse",
                    "constraints": {
                        "feature_limit": None,
                        "exclude_cols": ["product_id"]
                    }
                })
            },
            
            # Very casual/conversational prompt
            {
                "role": "user",
                "content": """Dataset: employee_data.csv (15 columns: emp_id, age, salary, dept, performance_score, years_experience, education, left_company)
User request: Can you help me figure out which employees might quit? We're losing too many good people."""
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "task": "classification",
                    "target": "left_company",
                    "metric": "recall",
                    "constraints": {
                        "feature_limit": None,
                        "exclude_cols": ["emp_id"],
                        "class_weight": "balanced"
                    }
                })
            },
            
            # Clustering request
            {
                "role": "user",
                "content": """Dataset: customer_behavior.csv (22 columns: cust_id, age, income, spending_category1, spending_category2, website_visits, etc.)
User request: Segment our customers into groups based on their behavior patterns for targeted marketing."""
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "task": "clustering",
                    "target": "",
                    "metric": "silhouette",
                    "constraints": {
                        "feature_limit": None,
                        "exclude_cols": ["cust_id"]
                    }
                })
            },
            
            # Risk assessment scenario
            {
                "role": "user",
                "content": """Dataset: loan_applications.csv (31 columns including applicant_id, credit_score, income, loan_amount, employment_years, default_risk)
User request: Build a quick model to assess loan default risk. We need high precision to avoid bad loans."""
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "task": "classification",
                    "target": "default_risk",
                    "metric": "precision",
                    "constraints": {
                        "feature_limit": 15,
                        "exclude_cols": ["applicant_id"],
                        "class_weight": "balanced",
                        "time_budget": "fast"
                    }
                })
            },
            
            # Pricing/valuation scenario
            {
                "role": "user",
                "content": """Dataset: real_estate.csv (28 columns: property_id, bedrooms, bathrooms, sqft, location, age, price, neighborhood_income)
User request: Create a comprehensive property valuation model. Accuracy is more important than speed."""
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "task": "regression",
                    "target": "price",
                    "metric": "rmse",
                    "constraints": {
                        "feature_limit": None,
                        "exclude_cols": ["property_id"],
                        "time_budget": "slow"
                    }
                })
            },
            
            # Complex scientific data
            {
                "role": "user",
                "content": """Dataset: complex_neural_dataset.csv (29 columns including sensor_reading_1, sensor_reading_2, experiment_condition, complex_target, lab_id)
User request: Predict complex target. Focus on accuracy."""
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "task": "classification",
                    "target": "complex_target",
                    "metric": "accuracy",
                    "constraints": {
                        "feature_limit": None,
                        "exclude_cols": ["lab_id"]
                    }
                })
            },
            
            # Medical/health scenario
            {
                "role": "user",
                "content": """Dataset: patient_data.csv (45 columns: patient_id, age, symptoms, test_results, medical_history, diagnosis)
User request: Help doctors predict disease risk. Missing a positive case is much worse than a false alarm."""
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "task": "classification",
                    "target": "diagnosis",
                    "metric": "recall",
                    "constraints": {
                        "feature_limit": 20,
                        "exclude_cols": ["patient_id"],
                        "class_weight": "balanced"
                    }
                })
            }
        ]
    def _prepare_context(self, request: PromptRequest) -> str:
        """Enhanced context preparation with richer dataset information
        
        This method prepares a detailed context string about a dataset to help GPT-4 understand 
        and interpret ML task requests. It analyzes the dataset preview and structures the information
        in a clear format.

        Args:
            request: PromptRequest object containing dataset preview info and user prompt

        The method:
        1. Extracts basic dataset info (columns, shape, dtypes, sample rows)
        2. Categorizes columns into numeric, categorical, ID-like, and date-like
        3. Creates a column summary showing the first 10 columns of each type
        4. Adds sample data showing first 3 rows (up to 8 columns each)
        5. Adds dataset size context (small/large) to guide model selection
        6. Formats everything into a clear prompt for GPT-4

        Returns:
            str: Formatted context string containing dataset analysis and user request
        """
        
        preview = request.dataset_preview or {}
        columns = preview.get("columns", [])
        shape = preview.get("shape", [0, 0])
        dtypes = preview.get("dtypes", {})
        sample_rows = preview.get("sample_rows", [])
        
        # Analyze column characteristics for better context
        numeric_cols = []
        categorical_cols = []
        id_like_cols = []
        date_like_cols = []
        
        for col in columns:
            col_lower = col.lower()
            if any(id_word in col_lower for id_word in ['id', '_id', 'key', 'index']):
                id_like_cols.append(col)
            elif any(date_word in col_lower for date_word in ['date', 'time', 'timestamp', 'created', 'updated']):
                date_like_cols.append(col)
            elif dtypes.get(col, '').startswith(('int', 'float')):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # Create column summary
        column_summary = f"Columns ({len(columns)} total):\n"
        
        if numeric_cols:
            column_summary += f"  Numeric: {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}\n"
        if categorical_cols:
            column_summary += f"  Categorical: {', '.join(categorical_cols[:10])}{'...' if len(categorical_cols) > 10 else ''}\n"
        if id_like_cols:
            column_summary += f"  ID-like: {', '.join(id_like_cols)}\n"
        if date_like_cols:
            column_summary += f"  Date-like: {', '.join(date_like_cols)}\n"
        
        # Add sample data context if available
        sample_context = ""
        if sample_rows and len(sample_rows) > 0:
            # Show first few rows to help understand data patterns
            sample_context = f"\nSample data (first {min(3, len(sample_rows))} rows):\n"
            for i, row in enumerate(sample_rows[:3]):
                row_preview = {k: v for k, v in list(row.items())[:8]}  # First 8 columns
                if len(row) > 8:
                    row_preview["..."] = f"({len(row)-8} more columns)"
                sample_context += f"  Row {i+1}: {row_preview}\n"
        
        # Dataset size context
        size_context = ""
        if shape[0] > 0:
            if shape[0] < 1000:
                size_context = " (small dataset - prefer simpler models)"
            elif shape[0] > 10000:
                size_context = " (large dataset - can use complex models)"
        
        context = f"""Dataset: {shape[0]:,} rows × {shape[1]} columns{size_context}

{column_summary}{sample_context}
User request: {request.prompt}

INSTRUCTIONS: Interpret this request as an ML task. Be flexible with business language and infer the most likely ML objective."""
        
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
    
    def _find_best_column_match(self, target_name: str, available_columns: List[str]) -> str:
        """Find the best matching column name from available columns.
        
        This method tries to find the closest matching column name from the available columns list
        by using several matching strategies in order:
        1. Exact match (case-insensitive)
        2. Normalized name match (removing special chars, spaces)
        3. Fuzzy string matching with similarity threshold
        
        Args:
            target_name: The target column name to find a match for
            available_columns: List of actual column names in the dataset
            
        Returns:
            str: The best matching column name from available_columns, or original target_name if no match found
        """
        
        if not available_columns:
            return target_name
        
        # Exact match (case-insensitive)
        for col in available_columns:
            if col.lower() == target_name.lower():
                return col
        
        # Normalize target name for matching
        normalized_target = self._normalize_column_name(target_name)
        
        # Try normalized matches
        for col in available_columns:
            normalized_col = self._normalize_column_name(col)
            if normalized_col == normalized_target:
                return col
        
        # Fuzzy matching for similar names
        best_match = None
        best_score = 0
        
        for col in available_columns:
            score = self._calculate_similarity(target_name, col)
            if score > best_score and score > 0.7:  # 70% similarity threshold
                best_score = score
                best_match = col
        
        if best_match:
            logger.info(f"Fuzzy matched '{target_name}' to '{best_match}' (score: {best_score:.2f})")
            return best_match
        
        # If no good match found, return original
        logger.warning(f"No good column match found for '{target_name}'")
        return target_name
    
    def _normalize_column_name(self, name: str) -> str:
        """Normalize column name for matching"""
        # Convert to lowercase
        normalized = name.lower()
        
        # Replace spaces and special chars with underscores
        normalized = re.sub(r'[^a-z0-9_]', '_', normalized)
        
        # Remove multiple underscores
        normalized = re.sub(r'_+', '_', normalized)
        
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        
        return normalized
    
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two column names"""
        # Simple similarity based on common substrings
        name1_norm = self._normalize_column_name(name1)
        name2_norm = self._normalize_column_name(name2)
        
        if name1_norm == name2_norm:
            return 1.0
        
        # Check if one contains the other
        if name1_norm in name2_norm or name2_norm in name1_norm:
            return 0.9
        
        # Check for word overlap
        words1 = set(name1_norm.split('_'))
        words2 = set(name2_norm.split('_'))
        
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            return overlap / total if total > 0 else 0.0
        
        return 0.0
    
    def _check_clarifications_needed(
        self, 
        spec: MLTaskSpec, 
        dataset_preview: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Check if clarifications are needed"""
        
        if not dataset_preview:
            return None
        
        columns = dataset_preview.get("columns", [])
        
        # Check if target column exists (after fuzzy matching)
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
                    # Proper null check - only count actual None/NaN/empty string values
                    null_count = 0
                    for row in sample_df:
                        value = row.get(col)
                        # Check for actual missing values (None, NaN, empty string)
                        if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
                            null_count += 1
                    
                    null_fraction = null_count / len(sample_df) if sample_df else 0
                    if null_fraction > 0.8:
                        issues.append(f"Column '{col}' has high null values ({null_fraction*100:.1f}%)")
        
        if issues:
            return "Data quality issues detected: " + "; ".join(issues) + ". Should these columns be excluded?"
        
        return None 