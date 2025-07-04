"""
GPT-powered query suggestion engine for Cipher datasets
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
            "sample_rows": [],  # Add actual sample rows for GPT analysis
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
        
        # Add sample rows for GPT to understand actual data patterns
        sample_size = min(8, len(df))  # Send up to 8 sample rows
        if sample_size > 0:
            # Get a representative sample (mix of first and random rows)
            if len(df) <= 8:
                sample_df = df.copy()
            else:
                # Take first 3 rows + 5 random rows for diversity
                first_rows = df.head(3)
                remaining_df = df.iloc[3:]
                random_rows = remaining_df.sample(n=min(5, len(remaining_df)), random_state=42)
                sample_df = pd.concat([first_rows, random_rows]).reset_index(drop=True)
            
            # Convert to list of dictionaries and clean for JSON serialization
            sample_rows = []
            for _, row in sample_df.iterrows():
                row_dict = {}
                for col in df.columns:
                    value = row[col]
                    # Handle different data types for JSON serialization
                    if pd.isna(value):
                        row_dict[col] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        row_dict[col] = float(value) if isinstance(value, np.floating) else int(value)
                    elif isinstance(value, (np.bool_, bool)):
                        row_dict[col] = bool(value)
                    else:
                        row_dict[col] = str(value)
                sample_rows.append(row_dict)
            
            analysis["sample_rows"] = sample_rows[:8]  # Ensure max 8 rows
        
        # Detect domain indicators based on column names (improved logic)
        column_names_lower = [col.lower() for col in df.columns]
        column_text = ' '.join(column_names_lower)
        
        # Technical/Scientific (check first to avoid false classifications)
        technical_indicators = ['sensor', 'quantum', 'electromagnetic', 'spectral', 'experiment', 'lab_condition', 'measurement_device', 'frequency', 'amplitude', 'phase', 'neural_activation', 'protein_expression', 'gene_activity', 'molecular', 'chemical', 'optical', 'acoustic', 'vibration']
        if any(indicator in column_text for indicator in technical_indicators):
            analysis["domain_indicators"].append("scientific")
        
        # Customer/Business
        elif any(keyword in column_text for keyword in [
            'customer', 'client', 'user', 'account', 'churn', 'tenure', 'revenue', 'spending', 'purchase', 'order'
        ]):
            analysis["domain_indicators"].append("business")
        
        # Health/medical (with exclusion for technical contexts)
        elif any(keyword in column_text for keyword in [
            'patient', 'age', 'blood', 'pressure', 'heart', 'disease', 'diagnosis', 'symptom', 'treatment', 'medical', 'clinical', 'hospital', 'doctor', 'therapy'
        ]):
            # Only classify as medical if no strong technical indicators
            has_technical = any(indicator in column_text for indicator in ['sensor', 'quantum', 'electromagnetic', 'spectral', 'experiment'])
            if not has_technical:
                analysis["domain_indicators"].append("healthcare")
        
        # Finance
        elif any(keyword in column_text for keyword in [
            'price', 'cost', 'revenue', 'profit', 'income', 'salary', 'loan', 'credit', 'financial'
        ]):
            analysis["domain_indicators"].append("finance")
        
        # HR/employee
        elif any(keyword in column_text for keyword in [
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
        sample_rows = dataset_analysis.get("sample_rows", [])
        
        # Format sample data for GPT analysis
        sample_data_text = ""
        if sample_rows:
            sample_data_text = f"""
Sample Data (first {len(sample_rows)} rows):
"""
            for i, row in enumerate(sample_rows[:5], 1):  # Show max 5 rows in prompt
                row_text = ", ".join([f"{k}: {v}" for k, v in list(row.items())[:10]])  # First 10 cols
                if len(row) > 10:
                    row_text += ", ..."
                sample_data_text += f"Row {i}: {row_text}\n"
        
        prompt = f"""You are an AI assistant helping users get started with machine learning on their dataset. 

Dataset Summary:
- Shape: {dataset_analysis['shape']['rows']} rows, {dataset_analysis['shape']['columns']} columns
- Potential target variables: {', '.join(potential_targets) if potential_targets else 'Not obvious'}
- Likely domain: {', '.join(domain_indicators) if domain_indicators else 'General'}

Columns:
{chr(10).join(columns_summary)}

{sample_data_text}

Based on the actual data patterns you can see above, generate {max_suggestions} diverse, actionable machine learning query suggestions that a user could ask about this dataset. 

COMPREHENSIVE ANALYSIS REQUIRED:
1. **Data Understanding**: Look at column names, data types, and sample values to understand the business domain
2. **Task Identification**: Identify what meaningful predictions or analyses could be performed
3. **Practical Value**: Focus on queries that would provide real business insights or actionable outcomes
4. **User-Friendly Language**: Write queries as natural business questions, not technical jargon
5. **Diverse Approaches**: Include different types of ML tasks (classification, regression, clustering where appropriate)

ANALYZE THE SAMPLE DATA to understand:
- What domain this data actually represents (healthcare, finance, scientific, customer behavior, etc.)
- What types of predictions would be meaningful and valuable
- What clustering approaches would provide business insights
- Which columns contain the most predictive or interesting information
- What business problems could be solved with this data

Each suggestion should:
1. Be a complete, natural question/request that a business user would ask
2. Specify what to predict or analyze (target variable or pattern to find)
3. Be appropriate for the ACTUAL data patterns and business domain you observe
4. Be ready to use as-is in a prompt
5. Provide clear business value and actionability

TASK TYPE GUIDANCE:
- **Classification**: When predicting categories, outcomes, or yes/no decisions
- **Regression**: When predicting numerical values, amounts, or continuous measures  
- **Clustering**: When discovering hidden patterns, customer segments, or grouping similar records

For clustering tasks, analyze the actual data characteristics and suggest specific, meaningful clustering use cases based on what you observe in the sample data. Consider what business insights could be gained from grouping similar records and how these groups could be used for decision-making.

BUSINESS CONTEXT EXAMPLES:
- E-commerce: "Predict which customers are likely to make a purchase in the next 30 days"
- Healthcare: "Identify patients at high risk for readmission based on their medical history"
- Finance: "Forecast monthly revenue based on historical trends and market indicators"
- HR: "Segment employees into groups based on performance and engagement patterns"
- Marketing: "Predict the optimal price for maximizing product sales"

Return your response as a JSON array of objects with this format:
[
  {{
    "query": "Predict customer churn risk to identify at-risk accounts for proactive retention",
    "type": "classification", 
    "target": "churn_flag",
    "description": "Binary classification to identify customers likely to cancel their subscription",
    "business_value": "Enable targeted retention campaigns and reduce customer acquisition costs",
    "actionable_insights": "Focus retention efforts on high-risk customers, understand key churn drivers"
  }}
]

Make the queries sound natural, business-focused, and directly actionable based on what you actually see in the data."""

        try:
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a machine learning expert helping users formulate good ML questions about their data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500  # Increased for more comprehensive suggestions
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
                        "description": suggestion.get("description", ""),
                        "business_value": suggestion.get("business_value", ""),
                        "actionable_insights": suggestion.get("actionable_insights", "")
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
        
        # Add intelligent clustering suggestions based on data
        if len(df) >= 10:  # Need enough data for clustering
            clustering_suggestion = self._generate_clustering_suggestion(df)
            if clustering_suggestion:
                suggestions.append(clustering_suggestion)
        
        # Add a general exploration suggestion
        if columns:
            suggestions.append({
                "query": f"Explore patterns and relationships in this {len(df)} row dataset",
                "type": "exploration",
                "target": "",
                "description": "General data exploration and pattern discovery"
            })
        
        return suggestions[:5]  # Return max 5 suggestions
    
    def _generate_clustering_suggestion(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate intelligent clustering suggestion based on data characteristics"""
        
        # Analyze data characteristics
        columns = df.columns.tolist()
        column_names_lower = [col.lower() for col in columns]
        
        # Detect domain and purpose
        domain_info = self._detect_clustering_domain(column_names_lower, df)
        
        if domain_info:
            return {
                "query": domain_info["query"],
                "type": "clustering", 
                "target": "",
                "description": domain_info["description"]
            }
        
        # Fallback generic clustering suggestion
        return {
            "query": f"Discover natural groups and patterns in this {len(df)} record dataset",
            "type": "clustering",
            "target": "",
            "description": "Use unsupervised learning to find hidden patterns and segment data into meaningful groups"
        }
    
    def _detect_clustering_domain(self, column_names_lower: List[str], df: pd.DataFrame) -> Optional[Dict[str, str]]:
        """Detect domain-specific clustering opportunities"""
        
        # Technical/Scientific/Research data (check first to avoid false medical classification)
        technical_indicators = ['sensor', 'quantum', 'electromagnetic', 'spectral', 'experiment', 'lab_condition', 'measurement_device', 'frequency', 'amplitude', 'phase', 'neural_activation', 'protein_expression', 'gene_activity', 'molecular', 'chemical', 'optical', 'acoustic', 'vibration']
        if any(indicator in ' '.join(column_names_lower) for indicator in technical_indicators):
            return {
                "query": f"Identify distinct experimental conditions and measurement patterns to understand underlying scientific phenomena",
                "description": "Scientific clustering for experiment classification, pattern discovery, and research optimization"
            }

        # Customer/Business data
        customer_indicators = ['customer', 'client', 'user', 'account', 'churn', 'tenure', 'revenue', 'spending', 'purchase', 'order']
        if any(indicator in ' '.join(column_names_lower) for indicator in customer_indicators):
            return {
                "query": f"Segment customers by behavior patterns, spending habits, and service usage to identify distinct customer groups",
                "description": "Customer segmentation for targeted marketing, retention strategies, and personalized service offerings"
            }
        
        # Healthcare/Medical data (be more specific to avoid false positives)
        health_indicators = ['patient', 'age', 'blood', 'pressure', 'heart', 'disease', 'diagnosis', 'symptom', 'treatment', 'medical', 'clinical', 'hospital', 'doctor', 'therapy']
        
        # Exclude technical/research contexts that might have medical-sounding terms
        technical_indicators = ['sensor', 'quantum', 'electromagnetic', 'spectral', 'experiment', 'lab_condition', 'measurement_device', 'frequency', 'amplitude', 'phase']
        column_text = ' '.join(column_names_lower)
        
        # Only classify as medical if it has medical indicators AND doesn't have strong technical indicators
        has_medical = any(indicator in column_text for indicator in health_indicators)
        has_technical = any(indicator in column_text for indicator in technical_indicators)
        
        if has_medical and not has_technical:
            return {
                "query": f"Group patients by symptom profiles, risk factors, and health characteristics to identify patient subpopulations",
                "description": "Patient stratification for personalized treatment plans and clinical decision support"
            }
        
        # Financial data
        finance_indicators = ['price', 'cost', 'income', 'salary', 'loan', 'credit', 'balance', 'transaction', 'payment', 'financial']
        if any(indicator in ' '.join(column_names_lower) for indicator in finance_indicators):
            return {
                "query": f"Identify distinct financial profiles and spending patterns to understand different financial behaviors",
                "description": "Financial segmentation for risk assessment, product recommendations, and investment strategies"
            }
        
        # Employee/HR data
        hr_indicators = ['employee', 'salary', 'department', 'performance', 'satisfaction', 'attrition', 'turnover', 'position', 'role']
        if any(indicator in ' '.join(column_names_lower) for indicator in hr_indicators):
            return {
                "query": f"Analyze employee profiles and performance patterns to identify distinct workforce segments",
                "description": "Employee segmentation for talent management, retention strategies, and organizational development"
            }
        
        # Product/Item data
        product_indicators = ['product', 'item', 'category', 'brand', 'rating', 'review', 'sales', 'inventory', 'sku']
        if any(indicator in ' '.join(column_names_lower) for indicator in product_indicators):
            return {
                "query": f"Group products by characteristics and performance metrics to identify product categories and market segments",
                "description": "Product segmentation for inventory management, pricing strategies, and market positioning"
            }
        
        # Geographic/Location data
        geo_indicators = ['location', 'city', 'state', 'country', 'region', 'address', 'latitude', 'longitude', 'zip', 'postal']
        if any(indicator in ' '.join(column_names_lower) for indicator in geo_indicators):
            return {
                "query": f"Discover geographic patterns and regional clusters to understand spatial relationships and local trends",
                "description": "Geographic segmentation for location-based services, market expansion, and regional analysis"
            }
        
        # Sensor/IoT data
        sensor_indicators = ['sensor', 'temperature', 'humidity', 'pressure', 'voltage', 'current', 'reading', 'measurement']
        if any(indicator in ' '.join(column_names_lower) for indicator in sensor_indicators):
            return {
                "query": f"Identify distinct operational patterns and anomalous behaviors in sensor readings and system metrics",
                "description": "Operational clustering for predictive maintenance, system optimization, and anomaly detection"
            }
        
        # Web/Digital data
        web_indicators = ['click', 'page', 'session', 'visit', 'bounce', 'conversion', 'engagement', 'traffic', 'user_id']
        if any(indicator in ' '.join(column_names_lower) for indicator in web_indicators):
            return {
                "query": f"Segment users by digital behavior patterns, engagement levels, and interaction preferences",
                "description": "User behavior segmentation for UX optimization, content personalization, and conversion improvement"
            }
        
        return None 