// Main application entry point for AutoML Desktop

class AutoMLApp {
    constructor() {
        this.isInitialized = false;
        this.backendHealthy = false;
    }

    // Initialize the application
    async init() {
        console.log('Initializing AutoML Desktop Application...');

        try {
            // Check if backend is available
            await this.waitForBackend();

            // Initialize API client
            apiClient.initSession();
            
            // Initialize UI manager
            uiManager.init();
            
            // Setup WebSocket connection
            await this.setupWebSocket();
            
            // Setup global event handlers
            this.setupGlobalHandlers();
            
            this.isInitialized = true;
            console.log('AutoML Desktop Application initialized successfully');
            
            uiManager.showNotification('success', 'Application Ready', 'AutoML Desktop is ready to use');
            
        } catch (error) {
            console.error('Failed to initialize application:', error);
            this.showBackendError(error);
        }
    }

    // Wait for backend to be available
    async waitForBackend(maxAttempts = 30, interval = 1000) {
        console.log('Waiting for backend to be available...');
        
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                const health = await apiClient.checkHealth();
                console.log('Backend health check successful:', health);
                this.backendHealthy = true;
                return health;
            } catch (error) {
                console.log(`Backend health check attempt ${attempt}/${maxAttempts} failed:`, error.message);
                
                if (attempt === maxAttempts) {
                    throw new Error('Backend is not available after maximum attempts');
                }
                
                // Wait before next attempt
                await new Promise(resolve => setTimeout(resolve, interval));
            }
        }
    }

    // Setup WebSocket connection and event handlers
    async setupWebSocket() {
        try {
            await apiClient.connectWebSocket();
            
            // Handle training progress updates
            apiClient.addEventListener('trial_update', (data) => {
                this.handleTrialUpdate(data);
            });
            
            // Handle family completion
            apiClient.addEventListener('family_complete', (data) => {
                this.handleFamilyComplete(data);
            });
            
            // Handle training completion
            apiClient.addEventListener('training_complete', (data) => {
                this.handleTrainingComplete(data);
            });
            
            // Handle errors
            apiClient.addEventListener('error', (data) => {
                this.handleError(data);
            });
            
            // Handle deployment completion
            apiClient.addEventListener('deployment_complete', (data) => {
                this.handleDeploymentComplete(data);
            });
            
            // Handle enhanced training events
            apiClient.addEventListener('ensemble_strategy', (data) => {
                uiManager.handleEnsembleStrategy(data);
            });
            
            apiClient.addEventListener('architecture_trial', (data) => {
                uiManager.handleArchitectureTrial(data);
            });
            
            apiClient.addEventListener('architecture_complete', (data) => {
                uiManager.handleArchitectureComplete(data);
            });
            
            // Handle training status updates
            apiClient.addEventListener('training_status', (data) => {
                this.handleTrainingStatus(data);
            });
            
            // Handle detailed PyTorch training events
            apiClient.addEventListener('training_start', (data) => {
                this.handleTrainingStart(data);
            });
            
            apiClient.addEventListener('epoch_update', (data) => {
                this.handleEpochUpdate(data);
            });
            
            apiClient.addEventListener('model_improvement', (data) => {
                this.handleModelImprovement(data);
            });
            
            apiClient.addEventListener('early_stopping', (data) => {
                this.handleEarlyStopping(data);
            });
            
            apiClient.addEventListener('pytorch_training_start', (data) => {
                this.handlePyTorchTrainingStart(data);
            });
            
            apiClient.addEventListener('pytorch_training_complete', (data) => {
                this.handlePyTorchTrainingComplete(data);
            });
            
        } catch (error) {
            console.warn('WebSocket connection failed:', error);
            uiManager.showNotification('warning', 'Connection Warning', 'Real-time updates may not be available');
        }
    }

    // Handle trial update from WebSocket
    handleTrialUpdate(data) {
        console.log('Trial update received:', data);
        
        // Add to training logs
        uiManager.addTrainingLog(data);
        
        // Update family progress
        uiManager.updateFamilyProgress(data.family, {
            status: 'running',
            trial: data.trial,
            val_metric: data.val_metric
        });
        
        // Update overall progress
        this.updateOverallProgress();
    }

    // Handle family completion
    handleFamilyComplete(data) {
        console.log('Family completed:', data);
        uiManager.updateFamilyProgress(data.family, {
            status: 'completed'
        });
    }

    // Handle training completion
    handleTrainingComplete(data) {
        console.log('Training completed:', data);
        
        // Check if this is enhanced training
        if (data.enhanced_results) {
            // Show enhanced training results
            uiManager.showEnhancedTrainingResults(data);
        } else {
            // Update ALL family statuses to completed for standard training
            const families = ['baseline', 'lightgbm', 'mlp'];
            families.forEach(family => {
                uiManager.updateFamilyProgress(family, {
                    status: 'completed'
                });
            });
            
            // Show standard training results
            uiManager.showTrainingResults(data);
        }
        
        // Update overall progress to 100%
        this.setOverallProgress(100);
    }

    // Handle errors from WebSocket
    handleError(data) {
        console.error('Error received from backend:', data);
        uiManager.showNotification('error', 'Training Error', data.message || 'An error occurred during training');
        uiManager.updateSessionStatus('error', 'Error occurred');
    }

    // Handle deployment completion
    handleDeploymentComplete(data) {
        console.log('Deployment completed:', data);
        uiManager.showDeploymentResults(data);
    }

    // Handle training status updates
    handleTrainingStatus(data) {
        console.log('Training status update:', data);
        uiManager.addTrainingStatusLog(data);
        
        // Handle model completion for enhanced mode
        if (data.data?.status === 'model_completed') {
            uiManager.handleModelCompletion(data);
        }
    }

    // Update overall progress based on family progress
    updateOverallProgress() {
        const families = ['baseline', 'lightgbm', 'mlp'];
        const totalFamilies = families.length;
        const maxTrialsPerFamily = 20; // From settings
        
        let totalProgress = 0;
        
        families.forEach(family => {
            const progress = uiManager.familyProgress[family];
            if (progress) {
                const familyProgress = Math.min(progress.trials / maxTrialsPerFamily, 1);
                totalProgress += familyProgress;
            }
        });
        
        const overallProgress = (totalProgress / totalFamilies) * 100;
        this.setOverallProgress(overallProgress);
    }

    // Set overall progress
    setOverallProgress(percentage) {
        const progressBar = document.getElementById('overallProgress');
        const progressText = document.getElementById('overallProgressText');
        
        if (progressBar && progressText) {
            progressBar.style.width = `${percentage}%`;
            progressText.textContent = `${Math.round(percentage)}%`;
        }
    }

    // Setup global event handlers
    setupGlobalHandlers() {
        // Handle window beforeunload
        window.addEventListener('beforeunload', () => {
            apiClient.disconnect();
        });
        
        // Handle focus/blur for connection management
        window.addEventListener('focus', () => {
            if (this.backendHealthy && !apiClient.websocket) {
                this.setupWebSocket().catch(console.error);
            }
        });
        
        // Handle keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });
        
        // Handle drag and drop on window
        window.addEventListener('dragover', (e) => {
            e.preventDefault();
        });
        
        window.addEventListener('drop', (e) => {
            e.preventDefault();
            
            // Only handle if we're on the data tab
            if (uiManager.currentTab === 'data') {
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    uiManager.handleFileSelection(files[0]);
                }
            }
        });
    }

    // Handle keyboard shortcuts
    handleKeyboardShortcuts(e) {
        // Ctrl/Cmd + N: New session
        if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
            e.preventDefault();
            uiManager.startNewSession();
        }
        
        // Ctrl/Cmd + Enter: Start training (if on data tab)
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && uiManager.currentTab === 'data') {
            e.preventDefault();
            const startBtn = document.getElementById('startTrainingBtn');
            if (!startBtn.disabled) {
                uiManager.startTraining();
            }
        }
        
        // Tab navigation with Ctrl/Cmd + 1-4
        if (e.ctrlKey || e.metaKey) {
            const tabKeys = {
                '1': 'data',
                '2': 'training',
                '3': 'results',
                '4': 'deployment'
            };
            
            if (tabKeys[e.key]) {
                e.preventDefault();
                uiManager.switchTab(tabKeys[e.key]);
            }
        }
    }

    // Show backend error
    showBackendError(error) {
        const errorMessage = `
            <div class="backend-error">
                <h2>Backend Connection Error</h2>
                <p>Failed to connect to the AutoML backend server.</p>
                <div class="error-details">
                    <strong>Error:</strong> ${error.message}
                </div>
                <div class="error-actions">
                    <button onclick="location.reload()" class="btn btn-primary">
                        <i class="fas fa-refresh"></i>
                        Retry Connection
                    </button>
                </div>
                <div class="error-help">
                    <h3>Troubleshooting:</h3>
                    <ul>
                        <li>Ensure Python and dependencies are installed</li>
                        <li>Check that the backend server is running</li>
                        <li>Verify the OPENAI_API_KEY is set (if required)</li>
                        <li>Try restarting the application</li>
                    </ul>
                </div>
            </div>
        `;
        
        document.querySelector('.content-area').innerHTML = errorMessage;
        uiManager.updateSessionStatus('error', 'Backend unavailable');
    }

    // Get application status
    getStatus() {
        return {
            initialized: this.isInitialized,
            backendHealthy: this.backendHealthy,
            sessionId: apiClient.sessionId,
            currentTab: uiManager.currentTab,
            currentRunId: uiManager.currentRunId,
            websocketConnected: apiClient.websocket && apiClient.websocket.readyState === WebSocket.OPEN
        };
    }

    // Handle training start event
    handleTrainingStart(data) {
        console.log('Training start received:', data);
        uiManager.addTrainingLog({
            family: data.model,
            trial: 0,
            val_metric: null,
            elapsed_s: 0,
            message: data.message
        });
    }

    // Handle epoch update event
    handleEpochUpdate(data) {
        console.log('Epoch update received:', data);
        
        // Add detailed epoch log
        uiManager.addEpochLog(data);
        
        // Update model progress if applicable
        if (data.model && data.progress) {
            uiManager.updateModelProgress(data.model, {
                progress: data.progress,
                val_score: data.val_score,
                epoch: data.epoch,
                total_epochs: data.total_epochs
            });
        }
    }

    // Handle model improvement event
    handleModelImprovement(data) {
        console.log('Model improvement received:', data);
        uiManager.addTrainingLog({
            family: data.model,
            trial: data.epoch,
            val_metric: data.new_best_score,
            elapsed_s: 0,
            message: data.message,
            type: 'improvement'
        });
    }

    // Handle early stopping event
    handleEarlyStopping(data) {
        console.log('Early stopping received:', data);
        uiManager.addTrainingLog({
            family: data.model,
            trial: data.stopped_at_epoch,
            val_metric: null,
            elapsed_s: 0,
            message: data.message,
            type: 'early_stop'
        });
    }

    // Handle PyTorch training start
    handlePyTorchTrainingStart(data) {
        console.log('PyTorch training start received:', data);
        uiManager.addTrainingLog({
            family: 'pytorch',
            trial: 0,
            val_metric: null,
            elapsed_s: 0,
            message: data.message,
            type: 'pytorch_start'
        });
    }

    // Handle PyTorch training complete
    handlePyTorchTrainingComplete(data) {
        console.log('PyTorch training complete received:', data);
        uiManager.addTrainingLog({
            family: 'pytorch',
            trial: data.models_trained || 0,
            val_metric: data.best_validation_score,
            elapsed_s: 0,
            message: data.message,
            type: 'pytorch_complete'
        });
    }

    // Cleanup resources
    cleanup() {
        apiClient.disconnect();
        console.log('AutoML Desktop Application cleaned up');
    }
}

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    console.log('DOM loaded, initializing AutoML Desktop...');
    
    // Create global app instance
    window.automlApp = new AutoMLApp();
    
    // Initialize the application
    await window.automlApp.init();
});

// Handle app closing
window.addEventListener('beforeunload', () => {
    if (window.automlApp) {
        window.automlApp.cleanup();
    }
}); 