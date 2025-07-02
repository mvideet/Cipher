# Architecture Details Feature

## Overview

This feature allows users to click into model family progress cards and view detailed information about each architecture tested within that family, along with their relative metrics in a beautiful UI.

## How It Works

### 1. Clickable Family Cards
- Family progress cards in the Training tab are now clickable
- Cards show a subtle hover effect with a hint text "Click to view architecture details"
- An expand arrow icon appears on hover to indicate interactivity

### 2. Architecture Details Modal
When you click on a family card, a modal opens showing:

- **Family Summary**: Overview statistics including:
  - Number of architectures tested
  - Best score achieved across all architectures
  - Current training status

- **Architecture Table**: Detailed table with columns for:
  - Architecture name and description
  - Current status (Pending, Running, Completed, Failed)
  - Number of trials completed
  - Best validation score achieved
  - Progress bar showing completion percentage
  - Training time elapsed

- **Performance Comparison**: Visual chart comparing scores across architectures

### 3. Architecture Data Tracking
The system now tracks individual architectures within each model family:

- **Enhanced Trainer Integration**: Captures architecture-level progress from the ML pipeline
- **Real-time Updates**: Architecture status and metrics update as training progresses
- **Persistent Storage**: Architecture data is maintained throughout the session

## Architecture Types

Different model families test various architectures:

### Random Forest / Tree-based Models
- **Fast Forest**: Quick training with moderate depth
- **Balanced Forest**: Balanced performance and speed  
- **Deep Forest**: Deep trees for complex patterns

### Gradient Boosting (XGBoost/LightGBM)
- **Fast Boost**: Fast boosting with early stopping
- **Balanced Boost**: Balanced boosting configuration
- **Precise Boost**: Slow but precise boosting

### Neural Networks (MLP)
- **Simple MLP**: Simple neural network
- **Balanced MLP**: Balanced neural architecture
- **Deep MLP**: Deep neural network for complex patterns

## Technical Implementation

### Frontend Components
1. **UI Manager Enhancements**: 
   - `familyArchitectures` data structure for tracking
   - `showArchitectureDetailsModal()` for displaying details
   - Helper methods for formatting and visualization

2. **CSS Styling**:
   - Modal overlay and responsive design
   - Architecture table with status badges
   - Progress bars and score visualization
   - Hover effects and animations

3. **Progress Tracking**:
   - `updateArchitectureProgress()` method
   - Integration with existing training logs
   - Real-time metric updates

### Data Structure
```javascript
familyArchitectures = {
  "lightgbm": {
    "fast_lgb": {
      name: "fast_lgb",
      trials: 3,
      bestScore: 0.847,
      status: "completed",
      metrics: [...],
      startTime: timestamp,
      completionTime: timestamp,
      config: {...}
    }
  }
}
```

## Usage

1. **Start Training**: Begin any ML training process (standard or enhanced mode)
2. **Navigate to Training Tab**: View the family progress cards
3. **Click Family Card**: Click on any family card (baseline, lightgbm, mlp, etc.)
4. **Explore Details**: View the detailed architecture breakdown in the modal
5. **Compare Performance**: Use the visual charts to compare architecture performance

## Benefits

- **Transparency**: See exactly which architectures are being tested
- **Performance Insights**: Understand which configurations work best for your data
- **Progress Monitoring**: Track individual architecture progress in real-time
- **Decision Making**: Make informed choices about which models to focus on

## Demo Data

For demonstration purposes, sample architecture data is populated when family progress is initialized, showing:
- LightGBM with 3 architectures (fast, balanced, precise)
- MLP with 3 architectures (simple, balanced, deep)  
- Baseline with 1 architecture (linear baseline)

This allows users to immediately see the feature in action and understand how it works. 