"""
GPT-powered query suggestion engine for AutoML datasets
"""

import structlog
import openai
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import json

from ..core.config import settings

logger = structlog.get_logger()


def convert_numpy_types(obj):
    """Convert numpy data types to native Python types for JSON serialization"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class QuerySuggester:
    """Generate intelligent query suggestions based on dataset analysis"""
    
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
    
    async def generate_suggestions(
        self, 
        df: pd.DataFrame, 
        max_suggestions: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate starter query suggestions based on dataset analysis"""
        
        if not self.client:
            logger.warning("OpenAI API key not available, using fallback suggestions")
            return self._generate_fallback_suggestions(df)
        
        try:
            # Analyze dataset structure
            dataset_analysis = self._analyze_dataset(df)
            
            # Generate suggestions using GPT
            suggestions = await self._generate_gpt_suggestions(dataset_analysis, max_suggestions)
            
            # Add fallback suggestions if GPT didn't generate enough
            if len(suggestions) < max_suggestions:
                fallback_suggestions = self._generate_fallback_suggestions(df)
                suggestions.extend(fallback_suggestions[:max_suggestions - len(suggestions)])
            
            logger.info(f"Generated {len(suggestions)} query suggestions")
            return suggestions[:max_suggestions]
            
        except Exception as e:
            logger.error("Failed to generate GPT suggestions, using fallback", error=str(e))
            return self._generate_fallback_suggestions(df)
    
    def _analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset structure for GPT context"""
        
        analysis = {
            "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
            "columns": [],
            "sample_data": {},
            "data_types": {},
            "potential_targets": [],
            "domain_indicators": []
        }
        
        # Analyze each column
        for col in df.columns:
            col_info = {
                "name": str(col),
                "type": str(df[col].dtype),
                "unique_values": int(min(df[col].nunique(), 20)),  # Cap at 20 for readability
                "null_count": int(df[col].isnull().sum()),
                "sample_values": []
            }
            
            # Get sample values (exclude nulls)
            sample_values = df[col].dropna().head(5).tolist()
            col_info["sample_values"] = [str(v) for v in sample_values]
            
            # Detect if this might be a target variable
            if col.lower() in ['target', 'label', 'class', 'outcome', 'result', 'prediction', 'y']:
                analysis["potential_targets"].append(str(col))
            elif df[col].nunique() <= 20 and df[col].dtype in ['int64', 'float64'] and col != 'id':
                analysis["potential_targets"].append(str(col))
            
            analysis["columns"].append(col_info)
        
        # Detect domain indicators based on column names
        column_names_lower = [col.lower() for col in df.columns]
        
        # Health/medical
        if any(keyword in ' '.join(column_names_lower) for keyword in [
            'age', 'blood', 'pressure', 'heart', 'disease', 'diagnosis', 'symptom', 'patient', 'medical'
        ]):
            analysis["domain_indicators"].append("healthcare")
        
        # Finance
        if any(keyword in ' '.join(column_names_lower) for keyword in [
            'price', 'cost', 'revenue', 'profit', 'income', 'salary', 'loan', 'credit', 'financial'
        ]):
            analysis["domain_indicators"].append("finance")
        
        # Sales/marketing
        if any(keyword in ' '.join(column_names_lower) for keyword in [
            'customer', 'sales', 'marketing', 'campaign', 'conversion', 'clicks', 'engagement'
        ]):
            analysis["domain_indicators"].append("marketing")
        
        # HR/employee
        if any(keyword in ' '.join(column_names_lower) for keyword in [
            'employee', 'salary', 'department', 'performance', 'satisfaction', 'attrition', 'turnover'
        ]):
            analysis["domain_indicators"].append("hr")
        
        return convert_numpy_types(analysis)
    
    async def _generate_gpt_suggestions(
        self, 
        dataset_analysis: Dict[str, Any], 
        max_suggestions: int
    ) -> List[Dict[str, Any]]:
        """Use GPT to generate contextual query suggestions"""
        
        # Prepare dataset summary for GPT
        columns_summary = []
        for col in dataset_analysis["columns"][:15]:  # Limit to prevent token overflow
            col_summary = f"- {col['name']} ({col['type']}): {col['unique_values']} unique values"
            if col["sample_values"]:
                sample_str = ", ".join(col["sample_values"][:3])
                col_summary += f", samples: {sample_str}"
            columns_summary.append(col_summary)
        
        potential_targets = dataset_analysis.get("potential_targets", [])
        domain_indicators = dataset_analysis.get("domain_indicators", [])
        
        prompt = f"""You are an AI assistant helping users get started with machine learning on their dataset. 

Dataset Summary:
- Shape: {dataset_analysis['shape']['rows']} rows, {dataset_analysis['shape']['columns']} columns
- Potential target variables: {', '.join(potential_targets) if potential_targets else 'Not obvious'}
- Likely domain: {', '.join(domain_indicators) if domain_indicators else 'General'}

Columns:
{chr(10).join(columns_summary)}

Generate {max_suggestions} diverse, actionable machine learning query suggestions that a user could ask about this dataset. Each suggestion should:
1. Be a complete, natural question/request
2. Specify what to predict (target variable)
3. Be appropriate for the data type and domain
4. Be ready to use as-is in a prompt

Focus on the most obvious and useful ML tasks for this data. Include different types of tasks (classification, regression, clustering if appropriate).

Return your response as a JSON array of objects with this format:
[
  {{
    "query": "Predict heart disease risk based on patient symptoms and test results",
    "type": "classification", 
    "target": "target",
    "description": "Binary classification to identify patients at risk"
  }}
]

Be concise but specific. Make the queries sound natural and actionable."""

        try:
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a machine learning expert helping users formulate good ML questions about their data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON if it's wrapped in markdown
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            suggestions = json.loads(response_text)
            
            # Validate and clean suggestions
            valid_suggestions = []
            for suggestion in suggestions:
                if isinstance(suggestion, dict) and "query" in suggestion:
                    # Ensure all required fields
                    clean_suggestion = {
                        "query": suggestion.get("query", "").strip(),
                        "type": suggestion.get("type", "classification").lower(),
                        "target": suggestion.get("target", ""),
                        "description": suggestion.get("description", "")
                    }
                    
                    if clean_suggestion["query"]:  # Must have a query
                        valid_suggestions.append(clean_suggestion)
            
            return valid_suggestions
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse GPT JSON response", error=str(e))
            return []
        except Exception as e:
            logger.error("GPT suggestion generation failed", error=str(e))
            return []
    
    def _generate_fallback_suggestions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate basic fallback suggestions when GPT is unavailable"""
        
        suggestions = []
        columns = df.columns.tolist()
        
        # Find potential target columns
        potential_targets = []
        for col in columns:
            if col.lower() in ['target', 'label', 'class', 'outcome', 'result', 'y']:
                potential_targets.append(col)
            elif df[col].nunique() <= 20 and df[col].dtype in ['int64', 'float64'] and col != 'id':
                potential_targets.append(col)
        
        # If no obvious targets, use the last column
        if not potential_targets and columns:
            potential_targets = [columns[-1]]
        
        # Generate basic suggestions
        for target in potential_targets[:3]:  # Max 3 targets
            # Determine task type based on target
            unique_values = df[target].nunique()
            task_type = "classification" if unique_values <= 20 else "regression"
            
            # Create basic query
            if task_type == "classification":
                query = f"Predict {target} using the other features in the dataset"
                description = f"Classification task to predict {target} categories"
            else:
                query = f"Predict the value of {target} based on other variables"
                description = f"Regression task to estimate {target} values"
            
            suggestions.append({
                "query": query,
                "type": task_type,
                "target": target,
                "description": description
            })
        
        # Add a general exploration suggestion
        if columns:
            suggestions.append({
                "query": f"Explore patterns and relationships in this {len(df)} row dataset",
                "type": "exploration",
                "target": "",
                "description": "General data exploration and pattern discovery"
            })
        
        return suggestions[:5]  # Return max 5 suggestions 