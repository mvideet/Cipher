# Cipher Desktop

A comprehensive desktop application for automated machine learning with a modern Electron UI and Python backend.

## Features

- **Intuitive Desktop Interface**: Modern Electron-based UI with drag-and-drop file upload
- **Automated ML Pipeline**: Upload CSV, describe your task in natural language, get trained models
- **GPT-4 Powered Prompt Parsing**: Intelligent extraction of ML task specifications from natural language
- **Multiple Model Families**: Automatic training and comparison of LightGBM, Neural Networks, and baseline models
- **Hyperparameter Optimization**: Optuna-powered automatic hyperparameter tuning
- **Model Explanability**: SHAP-based feature importance with AI-generated insights
- **One-Click Deployment**: Automatic Docker containerization of trained models
- **Real-Time Progress**: WebSocket-based live training progress updates
- **Audit Trail**: Complete logging and audit reports for reproducibility

## Quick Start

### Prerequisites

- **Python 3.11+** with pip
- **Node.js 18+** with npm
- **Docker** (for model deployment)
- **OpenAI API Key** (for prompt parsing and explanations)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cipher-desktop
   ```

2. **Set up environment variables**
   ```bash
   cp .env.template .env
   # Edit .env and add your OpenAI API key
   ```

3. **Install Python dependencies**
   ```bash
   pip install poetry
   poetry install
   ```

4. **Install Node.js dependencies**
   ```bash
   npm install
   ```

### Running the Application

1. **Development Mode** (recommended for first-time setup)
   ```bash
   npm run dev
   ```
   This will start both the Python backend and Electron frontend.

2. **Production Mode**
   ```bash
   # Start backend manually
   poetry run python -m uvicorn src.main:app --host 127.0.0.1 --port 8001
   
   # In another terminal, start Electron
   npm start
   ```

### Building Standalone Application

```bash
npm run build
```

This creates a standalone desktop application in the `dist/` folder.

## Usage Guide

### 1. Data Upload
- Click "Browse Files" or drag & drop a CSV file
- Supported: CSV files up to 100MB
- Preview shows first 10 rows and data statistics

### 2. Task Description
Describe your machine learning task in natural language. Examples:

- *"Predict customer churn. Optimize for recall. Use max 5 features. Exclude CustomerID."*
- *"Predict house prices using all features except property_id. Focus on RMSE."*
- *"Classify spam emails. Optimize F1 score. Exclude email_id and timestamp."*

### 3. Training
- Click "Start ML Pipeline"
- Watch real-time progress in the Training tab
- Multiple model families train in parallel
- Automatic hyperparameter optimization with Optuna

### 4. Results
- View best model performance
- Explore SHAP feature importance visualizations
- Read AI-generated insights about key factors

### 5. Deployment
- One-click Docker deployment
- Get ready-to-use API endpoints
- Download complete audit trail

## Architecture

### Backend (Python)
- **FastAPI**: REST API and WebSocket server
- **Optuna**: Hyperparameter optimization
- **LightGBM/Scikit-learn**: Machine learning models
- **SHAP**: Model explainability
- **SQLModel**: Database ORM
- **Docker**: Model containerization

### Frontend (Electron)
- **Electron**: Desktop application framework
- **Chart.js**: Data visualizations
- **Modern CSS**: Responsive, beautiful UI
- **WebSocket**: Real-time updates

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 | Required |
| `DEBUG` | Enable debug mode | `true` |
| `API_PORT` | Backend server port | `8001` |
| `MAX_TRAINING_TIME_MINUTES` | Training time limit | `15` |
| `MAX_OPTUNA_TRIALS` | Max trials per model family | `20` |
| `MAX_FILE_SIZE_MB` | File upload limit | `100` |

### Training Configuration

The application automatically:
- Splits data 80/20 for training/validation
- Handles missing values (median for numeric, mode for categorical)
- Applies feature scaling and encoding
- Performs feature selection if requested
- Uses cross-validation for small datasets (<500 rows)

### Model Families

1. **Baseline**: Logistic/Linear Regression with basic hyperparameters
2. **LightGBM**: Gradient boosting with extensive hyperparameter search
3. **MLP**: Neural networks with architecture and learning rate optimization

## Development

### Project Structure

```
cipher-desktop/
├── src/                    # Python backend
│   ├── api/               # FastAPI routes and WebSocket
│   ├── core/              # Configuration and utilities
│   ├── ml/                # ML pipeline components
│   └── models/            # Database and Pydantic models
├── app/                   # Electron frontend
│   ├── scripts/           # JavaScript modules
│   ├── styles/            # CSS stylesheets
│   ├── index.html         # Main UI
│   └── main.js            # Electron main process
├── pyproject.toml         # Python dependencies
└── package.json           # Node.js dependencies
```

### Adding New Model Families

1. Add model creation logic in `src/ml/trainer.py`
2. Define hyperparameter search space in `_suggest_params()`
3. Add model instantiation in `_create_model()`

### Extending the UI

1. Add new tabs in `app/index.html`
2. Style in `app/styles/main.css`
3. Add logic in `app/scripts/ui.js`

## Troubleshooting

### Backend Won't Start
- Check Python version: `python --version` (needs 3.11+)
- Install dependencies: `poetry install`
- Check OpenAI API key in `.env`

### Frontend Won't Connect
- Ensure backend is running on port 8001
- Check browser console for errors
- Verify WebSocket connection

### Training Fails
- Check dataset format (CSV with headers)
- Verify target column exists
- Ensure sufficient data (>50 rows recommended)

### Docker Deployment Fails
- Ensure Docker is running
- Check Docker permissions
- Verify model was trained successfully

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the application logs
3. Open an issue on GitHub with:
   - Steps to reproduce
   - Error messages
   - System information (OS, Python version, Node.js version)

## Recent Updates

### UI Fixes
- Adjusted the layout in the Enhanced Model Training section to remove unnecessary text and improve card positioning.

### Training Error Fix
- Resolved the error related to specifying columns using strings for non-DataFrame objects by ensuring proper column handling in the `_prepare_data` method.

### Ensemble Training
- Implemented ensemble training strategies such as voting and stacking to enhance model performance. This process involves creating an ensemble from trained models, leveraging their strengths for better predictions.

### Enhanced Model Training
- Utilizes ensemble strategies such as Voting and Stacking to improve model performance.
- Guided by the `ModelSelector` for recommending optimal models and strategies based on data characteristics.
- Supports a variety of model types, including neural networks, tailored to the specific task requirements.

---

**Note**: This application requires an OpenAI API key for optimal functionality. Without it, prompt parsing will be limited to basic pattern matching. 