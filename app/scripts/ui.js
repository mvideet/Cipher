// UI helper functions for AutoML Desktop

class UIManager {
    constructor() {
        this.currentTab = 'data';
        this.currentRunId = null;
        this.trainingStartTime = null;
        this.familyProgress = {};
    }

    // Initialize UI
    init() {
        this.setupTabNavigation();
        this.setupFileUpload();
        this.setupEventListeners();
        this.setupCopyButtons();
    }

    // Setup tab navigation
    setupTabNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        const tabContents = document.querySelectorAll('.tab-content');

        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const tabName = item.dataset.tab;
                this.switchTab(tabName);
            });
        });
    }

    // Switch between tabs
    switchTab(tabName) {
        // Update navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}Tab`).classList.add('active');

        this.currentTab = tabName;
    }

    // Setup file upload functionality
    setupFileUpload() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const browseBtn = document.getElementById('browseFileBtn');
        const removeBtn = document.getElementById('removeFileBtn');

        // Click to browse
        uploadArea.addEventListener('click', () => fileInput.click());
        browseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });

        // File selection
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.handleFileSelection(file);
            }
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelection(files[0]);
            }
        });

        // Remove file
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.clearFileSelection();
        });
    }

    // Handle file selection
    async handleFileSelection(file) {
        // Validate file
        const errors = validateCSVFile(file);
        if (errors.length > 0) {
            this.showNotification('error', 'Invalid File', errors.join(', '));
            return;
        }

        // Show file info
        this.showFileInfo(file);

        // Preview file content
        try {
            const text = await this.readFileAsText(file);
            const preview = parseCSVPreview(text);
            if (preview) {
                this.showDataPreview(preview);
            }
        } catch (error) {
            console.error('Failed to preview file:', error);
            this.showNotification('error', 'Preview Error', 'Failed to preview file content');
        }

        // Enable start button if prompt is also filled
        this.updateStartButtonState();
    }

    // Read file as text
    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = e => resolve(e.target.result);
            reader.onerror = reject;
            reader.readAsText(file);
        });
    }

    // Show file info
    showFileInfo(file) {
        const fileInfo = document.getElementById('fileInfo');
        const fileName = fileInfo.querySelector('.file-name');
        const fileStats = fileInfo.querySelector('.file-stats');

        fileName.textContent = file.name;
        fileStats.textContent = `${formatFileSize(file.size)} • Modified ${file.lastModified ? new Date(file.lastModified).toLocaleDateString() : 'Unknown'}`;

        document.getElementById('uploadArea').style.display = 'none';
        fileInfo.style.display = 'block';
    }

    // Clear file selection
    clearFileSelection() {
        document.getElementById('fileInput').value = '';
        document.getElementById('uploadArea').style.display = 'block';
        document.getElementById('fileInfo').style.display = 'none';
        document.getElementById('dataPreview').style.display = 'none';
        
        this.updateStartButtonState();
    }

    // Show data preview
    showDataPreview(preview) {
        const previewContainer = document.getElementById('dataPreview');
        const tableContainer = document.getElementById('previewTable');

        const tableHtml = createTable(preview);
        tableContainer.innerHTML = `
            <div class="preview-stats">
                <span><strong>Rows:</strong> ${preview.totalRows}</span>
                <span><strong>Columns:</strong> ${preview.totalColumns}</span>
                <span><strong>Showing:</strong> First ${preview.rows.length} rows</span>
            </div>
            <div class="table-wrapper">
                ${tableHtml}
            </div>
        `;

        previewContainer.style.display = 'block';
    }

    // Setup event listeners
    setupEventListeners() {
        // Start training button
        document.getElementById('startTrainingBtn').addEventListener('click', () => {
            this.startTraining();
        });

        // Deploy button
        document.getElementById('deployBtn').addEventListener('click', () => {
            this.deployModel();
        });

        // Download audit button
        document.getElementById('downloadAuditBtn').addEventListener('click', () => {
            this.downloadAuditReport();
        });

        // New session button
        document.getElementById('newSessionBtn').addEventListener('click', () => {
            this.startNewSession();
        });

        // Prompt input change
        document.getElementById('promptInput').addEventListener('input', () => {
            this.updateStartButtonState();
        });

        // Training mode change
        document.querySelectorAll('input[name="trainingMode"]').forEach(radio => {
            radio.addEventListener('change', () => {
                this.updateTrainingModeUI();
            });
        });
    }

    // Setup copy buttons
    setupCopyButtons() {
        document.addEventListener('click', async (e) => {
            if (e.target.matches('.copy-btn') || e.target.closest('.copy-btn')) {
                const btn = e.target.closest('.copy-btn');
                const targetSelector = btn.dataset.copyTarget;
                const targetElement = document.querySelector(targetSelector);
                
                if (targetElement) {
                    const text = targetElement.textContent.trim();
                    const success = await copyToClipboard(text);
                    
                    if (success) {
                        this.showNotification('success', 'Copied', 'Command copied to clipboard');
                    } else {
                        this.showNotification('error', 'Copy Failed', 'Failed to copy to clipboard');
                    }
                }
            }
        });
    }

    // Update start button state
    updateStartButtonState() {
        const fileSelected = document.getElementById('fileInfo').style.display !== 'none';
        const promptFilled = document.getElementById('promptInput').value.trim().length > 0;
        const startBtn = document.getElementById('startTrainingBtn');
        
        startBtn.disabled = !(fileSelected && promptFilled);
    }

    // Update training mode UI
    updateTrainingModeUI() {
        const selectedMode = document.querySelector('input[name="trainingMode"]:checked').value;
        const startBtnText = document.getElementById('startBtnText');
        
        if (selectedMode === 'enhanced') {
            startBtnText.textContent = 'Start Enhanced ML Pipeline';
        } else {
            startBtnText.textContent = 'Start ML Pipeline';
        }
    }

    // Start training
    async startTraining() {
        const fileInput = document.getElementById('fileInput');
        const promptInput = document.getElementById('promptInput');
        const selectedMode = document.querySelector('input[name="trainingMode"]:checked').value;
        
        const file = fileInput.files[0];
        const prompt = promptInput.value.trim();
        const enhanced = selectedMode === 'enhanced';
        
        // Debug logging for Enhanced Mode
        console.log('🔍 Training Mode Debug:', {
            selectedMode,
            enhanced,
            radioBtnValue: document.querySelector('input[name="trainingMode"]:checked')?.value,
            allRadioBtns: Array.from(document.querySelectorAll('input[name="trainingMode"]')).map(r => ({value: r.value, checked: r.checked}))
        });

        if (!file || !prompt) {
            this.showNotification('error', 'Missing Information', 'Please select a file and enter a prompt');
            return;
        }

        try {
            const loadingTitle = enhanced ? 'Getting AI Recommendations' : 'Starting ML Pipeline';
            const loadingSubtitle = enhanced ? 
                'AI is analyzing your data and recommending models...' : 
                'Uploading data and parsing prompt...';
            
            this.showLoading(loadingTitle, loadingSubtitle);
            
            // Use the new recommendation endpoint for enhanced mode
            console.log('📡 API Call Debug:', { enhanced, prompt: prompt.substring(0, 50) + '...' });
            const result = await apiClient.getModelRecommendations(file, prompt, enhanced);
            console.log('📥 API Response Debug:', { status: result.status, hasRecommendations: !!result.recommendations });
            this.hideLoading();
            
            if (result.status === 'needs_clarification') {
                console.log('💬 Showing clarification dialog');
                this.showClarificationDialog(result);
            } else if (result.status === 'recommendations_ready') {
                // Show model selection dialog for enhanced mode
                console.log('🎯 Showing Enhanced Mode model selection dialog');
                this.showModelSelectionDialog(
                    result.recommendations,
                    result.ensemble_strategy,
                    result.run_id,
                    result.session_id,
                    result.file_path,
                    result.parsed_intent
                );
            } else if (result.status === 'training_started') {
                // Direct to training for non-enhanced mode
                console.log('🚀 Starting training directly (Simple Mode)');
                this.currentRunId = result.run_id;
                this.handleTrainingStarted(result, enhanced);
            } else {
                console.log('❓ Unexpected status:', result.status);
            }
            
        } catch (error) {
            this.hideLoading();
            this.showNotification('error', 'Request Failed', handleError(error, 'Failed to process request'));
        }
    }

    // Handle training started
    handleTrainingStarted(result, enhanced = false) {
        this.trainingStartTime = Date.now();
        // Auto-detect enhanced mode from result data if not explicitly set
        if (!enhanced && (result.enhanced_features || result.recommendations || result.ensemble_strategy)) {
            enhanced = true;
        }
        this.isEnhancedMode = enhanced;
        
        // Update task configuration
        this.updateTaskConfiguration(result.parsed_intent);
        
        // Update data profile
        this.updateDataProfile(result.data_profile);
        
        // Switch to training tab
        this.switchTab('training');
        
        // Update session status
        const statusText = enhanced ? 'Enhanced training in progress' : 'Training in progress';
        this.updateSessionStatus('training', statusText);
        
        // Initialize family progress (different for enhanced mode)
        if (enhanced) {
            this.initializeEnhancedFamilyProgress();
        } else {
            this.initializeFamilyProgress();
        }
        
        // Clear training logs
        this.clearTrainingLogs();
        
        const notificationTitle = enhanced ? 'Enhanced Training Started' : 'Training Started';
        const notificationMessage = enhanced ? 
            'LLM-guided ensemble pipeline is now running with neural architecture search' : 
            'ML pipeline is now running';
        
        this.showNotification('success', notificationTitle, notificationMessage);
        
        // Show enhanced features if in enhanced mode
        if (enhanced && result.enhanced_features) {
            this.showEnhancedFeaturesNotification(result.enhanced_features);
        }
    }

    // Update task configuration
    updateTaskConfiguration(config) {
        document.getElementById('taskType').textContent = config.task;
        document.getElementById('targetColumn').textContent = config.target;
        document.getElementById('optimizationMetric').textContent = config.metric;
    }

    // Update data profile
    updateDataProfile(profile) {
        document.getElementById('dataRows').textContent = profile.n_rows.toLocaleString();
        document.getElementById('dataColumns').textContent = profile.n_cols;
        document.getElementById('dataIssues').textContent = profile.issues.length;
    }

    // Initialize family progress
    initializeFamilyProgress() {
        const families = ['baseline', 'lightgbm', 'mlp'];
        const container = document.getElementById('familyProgress');
        
        container.innerHTML = '';
        
        families.forEach(family => {
            this.familyProgress[family] = {
                status: 'pending',
                trials: 0,
                bestScore: null
            };
            
            const card = this.createFamilyCard(family);
            container.appendChild(card);
        });
    }

    // Create family progress card
    createFamilyCard(family) {
        const card = document.createElement('div');
        card.className = 'family-card';
        card.id = `family-${family}`;
        
        card.innerHTML = `
            <h4>
                ${family.charAt(0).toUpperCase() + family.slice(1)}
                <span class="family-status pending">Pending</span>
            </h4>
            <div class="family-details">
                <div class="family-stat">
                    <span class="label">Trials:</span>
                    <span class="value">0</span>
                </div>
                <div class="family-stat">
                    <span class="label">Best Score:</span>
                    <span class="value">-</span>
                </div>
            </div>
        `;
        
        return card;
    }

    // Update session status
    updateSessionStatus(status, text) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        
        statusText.textContent = text;
        
        // Update dot color based on status
        statusDot.className = 'status-dot';
        if (status === 'training') {
            statusDot.style.background = '#3b82f6';
        } else if (status === 'completed') {
            statusDot.style.background = '#10b981';
        } else if (status === 'error') {
            statusDot.style.background = '#ef4444';
        } else {
            statusDot.style.background = '#10b981';
        }
    }

    // Clear training logs
    clearTrainingLogs() {
        const logsContainer = document.getElementById('trainingLogs');
        logsContainer.innerHTML = '<div class="log-entries"></div>';
    }

    // Add training log entry
    addTrainingLog(entry) {
        const logsContainer = document.querySelector('#trainingLogs .log-entries');
        if (!logsContainer) {
            this.clearTrainingLogs();
            return this.addTrainingLog(entry);
        }
        
        const logElement = document.createElement('div');
        logElement.className = `log-entry ${entry.type || ''}`;
        
        const timestamp = formatTimestamp(new Date());
        const family = entry.family || entry.model || 'Unknown';
        const trial = entry.trial || 0;
        const metric = entry.val_metric !== undefined ? formatNumber(entry.val_metric) : '-';
        const elapsed = entry.elapsed_s || 0;
        
        // Handle different message types
        let content;
        if (entry.message && entry.type) {
            content = `
                <span class="log-timestamp">[${timestamp}]</span>
                <span class="log-message">${entry.message}</span>
            `;
        } else {
            content = `
                <span class="log-timestamp">[${timestamp}]</span>
                <span class="log-family">${family}</span>
                Trial ${trial}: <span class="log-metric">${metric}</span>
                (${formatDuration(elapsed)})
            `;
        }
        
        logElement.innerHTML = content;
        
        logsContainer.appendChild(logElement);
        logsContainer.scrollTop = logsContainer.scrollHeight;
        
        // Update enhanced model progress if in enhanced mode
        if (this.isEnhancedMode && family && trial) {
            this.updateEnhancedModelProgress(family, {
                trial: trial,
                val_metric: entry.val_metric
            });
        }
    }

    // Add detailed epoch log entry
    addEpochLog(epochData) {
        const logsContainer = document.querySelector('#trainingLogs .log-entries');
        if (!logsContainer) {
            this.clearTrainingLogs();
            return this.addEpochLog(epochData);
        }
        
        const logElement = document.createElement('div');
        logElement.className = 'log-entry epoch-log';
        
        const timestamp = formatTimestamp(new Date());
        const model = epochData.model || 'Neural Network';
        const progress = epochData.progress || 0;
        const epoch = epochData.epoch || 0;
        const totalEpochs = epochData.total_epochs || 0;
        const loss = epochData.loss ? epochData.loss.toFixed(4) : '-';
        const trainScore = epochData.train_score ? epochData.train_score.toFixed(4) : '-';
        const valScore = epochData.val_score ? epochData.val_score.toFixed(4) : '-';
        const lr = epochData.learning_rate || '-';
        const epochTime = epochData.epoch_time ? epochData.epoch_time.toFixed(2) + 's' : '-';
        
        logElement.innerHTML = `
            <span class="log-timestamp">[${timestamp}]</span>
            <span class="log-model">${model}</span>
            <div class="epoch-details">
                <span class="epoch-progress">Epoch ${epoch}/${totalEpochs} (${progress.toFixed(1)}%)</span>
                <span class="epoch-metrics">
                    Loss: ${loss} | Train: ${trainScore} | Val: ${valScore} | LR: ${lr} | Time: ${epochTime}
                </span>
            </div>
        `;
        
        logsContainer.appendChild(logElement);
        logsContainer.scrollTop = logsContainer.scrollHeight;
    }

    // Update model progress (for progress bars/indicators)
    updateModelProgress(modelName, progressData) {
        // Look for model progress indicator in the UI
        const modelProgressElement = document.querySelector(`[data-model="${modelName}"] .model-progress`);
        if (modelProgressElement) {
            const progressBar = modelProgressElement.querySelector('.progress-bar');
            const progressText = modelProgressElement.querySelector('.progress-text');
            
            if (progressBar) {
                progressBar.style.width = `${progressData.progress}%`;
            }
            
            if (progressText) {
                progressText.textContent = `${progressData.epoch}/${progressData.total_epochs} (${progressData.progress.toFixed(1)}%)`;
            }
            
            // Update validation score if element exists
            const valScoreElement = modelProgressElement.querySelector('.val-score');
            if (valScoreElement && progressData.val_score !== undefined) {
                valScoreElement.textContent = progressData.val_score.toFixed(4);
            }
        }
    }

    // Add training status log entry
    addTrainingStatusLog(statusData) {
        const logsContainer = document.querySelector('#trainingLogs .log-entries');
        if (!logsContainer) {
            this.clearTrainingLogs();
            return this.addTrainingStatusLog(statusData);
        }
        
        const logElement = document.createElement('div');
        logElement.className = 'log-entry status-log';
        
        const timestamp = formatTimestamp(new Date());
        
        // Extract message from nested data structure
        let message = 'Status update';
        if (typeof statusData.data?.message === 'string') {
            message = statusData.data.message;
        } else if (typeof statusData.data?.message === 'object') {
            message = statusData.data.message.message || JSON.stringify(statusData.data.message);
        } else if (typeof statusData.message === 'string') {
            message = statusData.message;
        }
        
        const status = statusData.data?.status || statusData.status || 'info';
        
        logElement.innerHTML = `
            <span class="log-timestamp">[${timestamp}]</span>
            <span class="log-status">${status.toUpperCase()}</span>
            <span class="log-message">${message}</span>
        `;
        
        logsContainer.appendChild(logElement);
        logsContainer.scrollTop = logsContainer.scrollHeight;
    }

    // Update family progress
    updateFamilyProgress(family, data) {
        this.familyProgress[family] = {
            ...this.familyProgress[family],
            ...data
        };
        
        const card = document.getElementById(`family-${family}`);
        if (!card) return;
        
        const statusElement = card.querySelector('.family-status');
        const trialsElement = card.querySelector('.family-stat:nth-child(1) .value');
        const scoreElement = card.querySelector('.family-stat:nth-child(2) .value');
        
        statusElement.textContent = data.status || this.familyProgress[family].status;
        statusElement.className = `family-status ${data.status || this.familyProgress[family].status}`;
        
        if (data.trial) {
            trialsElement.textContent = data.trial;
        }
        
        if (data.val_metric !== undefined) {
            scoreElement.textContent = formatNumber(data.val_metric);
        }
    }

    // Show training results
    showTrainingResults(results) {
        // Hide placeholder
        document.getElementById('resultsContent').style.display = 'none';
        
        // Show results
        const resultsContainer = document.getElementById('modelResults');
        resultsContainer.style.display = 'block';
        
        // Update best model info
        const bestModel = results.best_model;
        document.getElementById('bestFamily').textContent = bestModel.family;
        document.getElementById('bestScore').textContent = formatNumber(bestModel.val_score);
        document.getElementById('trainScore').textContent = formatNumber(bestModel.train_score);
        
        // Show explanations
        if (results.explanation) {
            this.showExplanations(results.explanation);
        }
        
        // Enable deployment
        document.getElementById('deployBtn').disabled = false;
        
        // Switch to results tab
        this.switchTab('results');
        
        // Update session status
        this.updateSessionStatus('completed', 'Training completed');
        
        this.showNotification('success', 'Training Complete', 'Model training has finished successfully');
    }

    // Show explanations
    showExplanations(explanation) {
        // Feature importance chart
        if (explanation.feature_importance) {
            this.createFeatureImportanceChart(explanation.feature_importance);
        }
        
        // Text explanation
        if (explanation.text_explanation) {
            document.getElementById('textExplanation').textContent = explanation.text_explanation;
        }
    }

    // Create feature importance chart
    createFeatureImportanceChart(featureImportance) {
        const ctx = document.getElementById('featureImportanceChart');
        if (!ctx) return;
        
        // Clear existing content
        ctx.innerHTML = '';
        
        // Create canvas
        const canvas = document.createElement('canvas');
        ctx.appendChild(canvas);
        
        const features = Object.keys(featureImportance).slice(0, 10); // Top 10
        const values = features.map(f => featureImportance[f]);
        
        new Chart(canvas, {
            type: 'bar',
            data: {
                labels: features,
                datasets: [{
                    label: 'Feature Importance',
                    data: values,
                    backgroundColor: features.map((_, i) => getChartColor(i)),
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Importance Score'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Features'
                        }
                    }
                }
            }
        });
    }

    // Deploy model
    async deployModel() {
        if (!this.currentRunId) {
            this.showNotification('error', 'No Model', 'No trained model available for deployment');
            return;
        }

        try {
            this.showLoading('Deploying Model', 'Building Docker container...');
            
            const result = await apiClient.deployModel(this.currentRunId);
            
            this.hideLoading();
            this.showDeploymentResults(result);
            
        } catch (error) {
            this.hideLoading();
            this.showNotification('error', 'Deployment Failed', handleError(error, 'Failed to deploy model'));
        }
    }

    // Show deployment results
    showDeploymentResults(result) {
        // Hide placeholder
        document.getElementById('deploymentContent').style.display = 'none';
        
        // Show results
        const resultsContainer = document.getElementById('deploymentResults');
        resultsContainer.style.display = 'block';
        
        // Update deployment info
        document.getElementById('dockerTag').textContent = result.docker_tag;
        document.getElementById('imageSize').textContent = formatFileSize(result.image_size_mb * 1024 * 1024);
        document.getElementById('buildTime').textContent = formatDuration(result.build_time_seconds);
        document.getElementById('deploymentCommand').textContent = result.deployment_command;
        
        // Switch to deployment tab
        this.switchTab('deployment');
        
        this.showNotification('success', 'Deployment Complete', 'Model has been deployed successfully');
    }

    // Download audit report
    async downloadAuditReport() {
        if (!this.currentRunId) {
            this.showNotification('error', 'No Run', 'No run available for audit report');
            return;
        }

        try {
            // In a real implementation, this would download the audit file
            // For now, we'll create a simple report
            const report = {
                runId: this.currentRunId,
                timestamp: new Date().toISOString(),
                status: 'completed'
            };
            
            const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `automl-audit-${this.currentRunId}.json`;
            a.click();
            
            URL.revokeObjectURL(url);
            
            this.showNotification('success', 'Download Started', 'Audit report is being downloaded');
            
        } catch (error) {
            this.showNotification('error', 'Download Failed', handleError(error, 'Failed to download audit report'));
        }
    }

    // Start new session
    startNewSession() {
        if (confirm('Starting a new session will clear all current data. Continue?')) {
            location.reload();
        }
    }

    // Show loading overlay
    showLoading(title, subtitle) {
        document.getElementById('loadingText').textContent = title;
        document.getElementById('loadingSubtext').textContent = subtitle;
        document.getElementById('loadingOverlay').style.display = 'flex';
    }

    // Hide loading overlay
    hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }

    // Show notification
    showNotification(type, title, message, duration = 5000) {
        const container = document.getElementById('notifications');
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        notification.innerHTML = `
            <div class="notification-header">
                <span class="notification-title">${title}</span>
            </div>
            <div class="notification-message">${message}</div>
        `;
        
        container.appendChild(notification);
        
        // Auto remove after duration
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, duration);
        
        // Click to dismiss
        notification.addEventListener('click', () => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        });
    }

    // Show clarification dialog
    showClarificationDialog(result) {
        const dialog = document.createElement('div');
        dialog.className = 'clarification-overlay';
        dialog.innerHTML = `
            <div class="clarification-dialog">
                <div class="clarification-header">
                    <h2><i class="fas fa-question-circle"></i> Clarification Needed</h2>
                    <p>The AI needs more information to proceed with training.</p>
                </div>
                
                <div class="clarification-content">
                    <div class="clarification-question">
                        <label>Question:</label>
                        <div class="question-text">${result.clarifications}</div>
                    </div>
                    
                    <div class="clarification-input">
                        <label for="clarificationText">Your Response:</label>
                        <textarea 
                            id="clarificationText" 
                            placeholder="Please provide the requested clarification..."
                            rows="4"
                        ></textarea>
                    </div>
                </div>
                
                <div class="clarification-footer">
                    <button class="btn btn-secondary" id="cancelClarification">
                        Cancel
                    </button>
                    <button class="btn btn-primary" id="submitClarification">
                        <i class="fas fa-paper-plane"></i>
                        Submit Clarification
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(dialog);
        
        // Focus on textarea
        const textarea = dialog.querySelector('#clarificationText');
        textarea.focus();
        
        // Setup event listeners
        dialog.querySelector('#cancelClarification').addEventListener('click', () => {
            dialog.remove();
        });
        
        dialog.querySelector('#submitClarification').addEventListener('click', () => {
            const clarification = textarea.value.trim();
            if (clarification) {
                this.provideClarification(result.run_id, clarification);
                dialog.remove();
            } else {
                this.showNotification('warning', 'Input Required', 'Please provide a clarification before submitting.');
            }
        });
        
        // Handle Enter key in textarea (with Ctrl/Cmd + Enter to submit)
        textarea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                dialog.querySelector('#submitClarification').click();
            }
        });
        
        // Handle ESC key to close
        dialog.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                dialog.remove();
            }
        });
    }

    // Provide clarification
    async provideClarification(runId, clarification) {
        try {
            this.showLoading('Processing Clarification', 'Continuing with training...');
            
            const result = await apiClient.provideClarification(runId, clarification);
            
            this.hideLoading();
            
            if (result.training_resumed) {
                this.showNotification('success', 'Clarification Received', 'Training is now resuming');
            }
            
        } catch (error) {
            this.hideLoading();
            this.showNotification('error', 'Clarification Failed', handleError(error, 'Failed to process clarification'));
        }
    }

    // Initialize enhanced family progress
    initializeEnhancedFamilyProgress() {
        const container = document.getElementById('familyProgress');
        container.innerHTML = '<div class="enhanced-progress-placeholder"><i class="fas fa-brain"></i><p>Waiting for AI model recommendations...</p></div>';
        this.familyProgress = {};
    }

    // Show enhanced features notification
    showEnhancedFeaturesNotification(features) {
        const featureList = features.map(f => `• ${f}`).join('\n');
        this.showNotification('info', 'Enhanced Features Active', featureList, 8000);
    }

    // Handle ensemble strategy update (enhanced mode)
    handleEnsembleStrategy(data) {
        if (!this.isEnhancedMode) return;

        const container = document.getElementById('familyProgress');
        
        // Update with actual model recommendations
        container.innerHTML = `
            <div class="ensemble-strategy">
                <h4><i class="fas fa-brain"></i> AI-Selected Ensemble Strategy</h4>
                <div class="strategy-info">
                    <div class="strategy-item">
                        <span class="label">Method:</span>
                        <span class="value">${data.strategy}</span>
                    </div>
                    <div class="strategy-item">
                        <span class="label">Selected Models:</span>
                        <span class="value">${data.models.join(', ')}</span>
                    </div>
                </div>
                <div class="strategy-reasoning">
                    <h5>AI Reasoning:</h5>
                    <p>${data.reasoning}</p>
                </div>
            </div>
            <div id="enhancedModelProgress" class="enhanced-model-grid"></div>
        `;

        this.addTrainingLog({
            family: 'AI-Advisor',
            trial: 1,
            val_metric: `Strategy: ${data.strategy}`,
            elapsed_s: (Date.now() - this.trainingStartTime) / 1000
        });
    }

    // Handle architecture trial update (enhanced mode)
    handleArchitectureTrial(data) {
        if (!this.isEnhancedMode) return;

        // Update specific model architecture progress
        this.updateEnhancedModelProgress(data.model, data);

        // Add to training logs
        this.addTrainingLog({
            family: data.model,
            trial: data.trial,
            val_metric: data.val_metric,
            elapsed_s: data.elapsed_s
        });
    }

    // Update enhanced model progress
    updateEnhancedModelProgress(modelName, data) {
        let container = document.getElementById('enhancedModelProgress');
        
        // If no enhanced progress container, create one
        if (!container) {
            const familyProgress = document.getElementById('familyProgress');
            if (familyProgress && this.isEnhancedMode) {
                familyProgress.innerHTML = `
                    <div class="ensemble-strategy">
                        <h4><i class="fas fa-brain"></i> Enhanced Training Progress</h4>
                        <p>Multiple model architectures are being trained and optimized.</p>
                    </div>
                    <div id="enhancedModelProgress" class="enhanced-model-grid"></div>
                `;
                container = document.getElementById('enhancedModelProgress');
            }
        }
        
        if (!container) return;

        let modelCard = document.getElementById(`enhanced-${modelName}`);
        
        if (!modelCard) {
            modelCard = document.createElement('div');
            modelCard.id = `enhanced-${modelName}`;
            modelCard.className = 'enhanced-model-card';
            modelCard.innerHTML = `
                <h5>${modelName}</h5>
                <div class="enhanced-model-details">
                    <div class="enhanced-stat">
                        <span class="label">Architectures Tested:</span>
                        <span class="value">0</span>
                    </div>
                    <div class="enhanced-stat">
                        <span class="label">Best Score:</span>
                        <span class="value">-</span>
                    </div>
                    <div class="enhanced-stat">
                        <span class="label">Status:</span>
                        <span class="status running">Training</span>
                    </div>
                </div>
            `;
            container.appendChild(modelCard);
        }

        // Update values
        const trialsElement = modelCard.querySelector('.enhanced-stat:nth-child(1) .value');
        const scoreElement = modelCard.querySelector('.enhanced-stat:nth-child(2) .value');

        if (data.trial) {
            trialsElement.textContent = data.trial;
        }

        if (data.val_metric !== undefined) {
            scoreElement.textContent = formatNumber(data.val_metric);
        }
    }

    // Handle architecture completion (enhanced mode)
    handleArchitectureComplete(data) {
        if (!this.isEnhancedMode) return;

        const modelCard = document.getElementById(`enhanced-${data.model}`);
        if (modelCard) {
            const statusElement = modelCard.querySelector('.status');
            statusElement.textContent = 'Completed';
            statusElement.className = 'status completed';
        }

        this.addTrainingLog({
            family: data.model,
            trial: 'Final',
            val_metric: data.val_metric,
            elapsed_s: data.elapsed_s
        });
    }

    // Handle model completion status updates
    handleModelCompletion(data) {
        if (this.isEnhancedMode) {
            // For enhanced mode, mark architecture as completed
            const allCards = document.querySelectorAll('.enhanced-model-card .status');
            allCards.forEach(statusElement => {
                if (statusElement.textContent === 'Training') {
                    statusElement.textContent = 'Completed';
                    statusElement.className = 'status completed';
                }
            });
        }
    }

    // Handle enhanced training results
    showEnhancedTrainingResults(results) {
        // Show enhanced results with ensemble information
        this.showTrainingResults(results);

        // Add enhanced-specific information
        if (results.enhanced_results) {
            const enhancedInfo = document.createElement('div');
            enhancedInfo.className = 'enhanced-results-info';
            enhancedInfo.innerHTML = `
                <h4><i class="fas fa-brain"></i> Enhanced Training Results</h4>
                <div class="enhanced-details">
                    <div class="enhanced-detail">
                        <span class="label">Ensemble Method:</span>
                        <span class="value">${results.enhanced_results.ensemble_method}</span>
                    </div>
                    <div class="enhanced-detail">
                        <span class="label">Models Tested:</span>
                        <span class="value">${results.enhanced_results.models_tested}</span>
                    </div>
                    <div class="enhanced-detail">
                        <span class="label">Selection Method:</span>
                        <span class="value">${results.enhanced_results.selection_method}</span>
                    </div>
                </div>
            `;

            const resultsContainer = document.getElementById('modelResults');
            resultsContainer.insertBefore(enhancedInfo, resultsContainer.firstChild);
        }
    }

    // Show model selection dialog
    showModelSelectionDialog(recommendations, ensemble_strategy, run_id, session_id, file_path, parsed_intent) {
        console.log('🎯 Model Selection Dialog Called:', {
            recommendationsCount: recommendations.length,
            models: recommendations.map(r => r.model_type),
            ensembleMethod: ensemble_strategy.method
        });
        const dialog = document.createElement('div');
        dialog.className = 'model-selection-overlay';
        dialog.innerHTML = `
            <div class="model-selection-dialog">
                <div class="model-selection-header">
                    <h2><i class="fas fa-brain"></i> AI Model Recommendations</h2>
                    <p>Our AI has analyzed your data and recommends these models. Select which ones to train:</p>
                    <div class="ensemble-info">
                        <strong>Ensemble Strategy:</strong> ${ensemble_strategy.method}
                        <br><small>${ensemble_strategy.reasoning}</small>
                    </div>
                </div>
                
                <div class="model-selection-content">
                    <div class="model-grid">
                        ${recommendations.map(model => this.createModelCard(model)).join('')}
                    </div>
                </div>
                
                <div class="model-selection-footer">
                    <div class="selection-summary">
                        <span id="selectedCount">${recommendations.filter(m => m.selected).length}</span> of ${recommendations.length} models selected
                        <br><small>Estimated training time: <span id="estimatedTime">15-45 minutes</span></small>
                    </div>

                    <div class="selection-actions">
                        <button class="btn btn-secondary" onclick="this.closeModelSelection()">
                            Cancel
                        </button>
                        <button class="btn btn-primary" id="startSelectedTraining">
                            <i class="fas fa-play"></i>
                            Start Training Selected Models
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(dialog);
        
        // Setup event listeners
        this.setupModelSelectionListeners(dialog, recommendations, run_id, session_id, file_path, parsed_intent);
    }

    // Create model card for selection
    createModelCard(model) {
        const complexityColor = model.complexity_score <= 3 ? '#10b981' : 
                              model.complexity_score <= 7 ? '#f59e0b' : '#ef4444';
        
        const interpretabilityIcon = model.interpretability === 'high' ? 'eye' : 
                                   model.interpretability === 'medium' ? 'eye-slash' : 'lock';
        
        return `
            <div class="model-card ${model.selected ? 'selected' : ''}" data-model-id="${model.id}">
                <div class="model-card-header">
                    <div class="model-selection-checkbox">
                        <input type="checkbox" id="model_${model.id}" ${model.selected ? 'checked' : ''}>
                        <label for="model_${model.id}"></label>
                    </div>
                    <div class="model-info">
                        <h3>${model.name}</h3>
                        <span class="model-family">${model.model_family.replace('_', ' ')}</span>
                    </div>
                    <div class="model-metrics">
                        <div class="metric">
                            <span class="metric-label">Complexity</span>
                            <div class="complexity-bar">
                                <div class="complexity-fill" style="width: ${model.complexity_score * 10}%; background: ${complexityColor}"></div>
                            </div>
                            <span class="metric-value">${model.complexity_score}/10</span>
                        </div>
                    </div>
                </div>
                
                <div class="model-card-body">
                    <div class="model-attributes">
                        <div class="attribute">
                            <i class="fas fa-clock"></i>
                            <span>Training: ${model.expected_training_time}</span>
                        </div>
                        <div class="attribute">
                            <i class="fas fa-memory"></i>
                            <span>Memory: ${model.memory_usage}</span>
                        </div>
                        <div class="attribute">
                            <i class="fas fa-${interpretabilityIcon}"></i>
                            <span>Interpretability: ${model.interpretability}</span>
                        </div>
                        <div class="attribute">
                            <i class="fas fa-cogs"></i>
                            <span>${model.architectures} architectures</span>
                        </div>
                    </div>
                    
                    <div class="model-description">
                        <p><strong>Best for:</strong> ${model.best_for}</p>
                        <p class="reasoning">${model.reasoning}</p>
                    </div>
                    
                    <div class="pros-cons">
                        <div class="pros">
                            <h4><i class="fas fa-check-circle"></i> Pros</h4>
                            <ul>
                                ${model.pros.map(pro => `<li>${pro}</li>`).join('')}
                            </ul>
                        </div>
                        <div class="cons">
                            <h4><i class="fas fa-exclamation-triangle"></i> Cons</h4>
                            <ul>
                                ${model.cons.map(con => `<li>${con}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Setup model selection event listeners
    setupModelSelectionListeners(dialog, recommendations, run_id, session_id, file_path, parsed_intent) {
        // Model selection checkboxes
        dialog.querySelectorAll('.model-selection-checkbox input').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const modelId = e.target.id.replace('model_', '');
                const modelCard = dialog.querySelector(`[data-model-id="${modelId}"]`);
                const model = recommendations.find(m => m.id === modelId);
                
                if (e.target.checked) {
                    modelCard.classList.add('selected');
                    model.selected = true;
                } else {
                    modelCard.classList.remove('selected');
                    model.selected = false;
                }
                
                this.updateSelectionSummary(dialog, recommendations);
            });
        });
        
        // Start training button
        dialog.querySelector('#startSelectedTraining').addEventListener('click', () => {
            this.startTrainingWithSelection(recommendations, run_id, session_id, file_path, parsed_intent);
            dialog.remove();
        });
        
        // Cancel button
        dialog.querySelector('.btn-secondary').addEventListener('click', () => {
            dialog.remove();
        });
    }

    // Update selection summary
    updateSelectionSummary(dialog, recommendations) {
        const selectedModels = recommendations.filter(m => m.selected);
        const selectedCount = dialog.querySelector('#selectedCount');
        const estimatedTime = dialog.querySelector('#estimatedTime');
        const startButton = dialog.querySelector('#startSelectedTraining');
        
        selectedCount.textContent = selectedModels.length;
        
        // Estimate time based on selected models
        let totalTime = 0;
        selectedModels.forEach(model => {
            const timeMultiplier = {
                'fast': 5,
                'medium': 15,
                'slow': 30
            };
            totalTime += timeMultiplier[model.expected_training_time] || 15;
        });
        
        estimatedTime.textContent = selectedModels.length === 0 ? '0 minutes' : 
                                   `${Math.round(totalTime * 0.7)}-${Math.round(totalTime * 1.3)} minutes`;
        
        startButton.disabled = selectedModels.length === 0;
    }

    // Start training with selected models
    async startTrainingWithSelection(recommendations, run_id, session_id, file_path, parsed_intent) {
        const selectedModels = recommendations.filter(m => m.selected).map(m => m.id);
        
        if (selectedModels.length === 0) {
            this.showNotification('error', 'No Models Selected', 'Please select at least one model to train');
            return;
        }
        
        try {
            this.showLoading('Starting Training', `Training ${selectedModels.length} selected models...`);
            
            const result = await apiClient.startTrainingWithSelection({
                run_id,
                session_id,
                selected_models: selectedModels,
                file_path,
                parsed_intent
            });
            
            this.currentRunId = run_id;
            this.hideLoading();
            
            // Switch to training view
            this.handleTrainingStarted({
                ...result,
                parsed_intent,
                data_profile: { n_rows: 0, n_cols: 0, issues: [] } // Will be updated via websocket
            }, true);
            
            this.showNotification('success', 'Training Started', 
                                `Training started with ${selectedModels.length} selected models`);
            
        } catch (error) {
            this.hideLoading();
            this.showNotification('error', 'Training Failed', 
                                handleError(error, 'Failed to start training with selected models'));
        }
    }

    // Close model selection
    closeModelSelection() {
        const dialog = document.querySelector('.model-selection-overlay');
        if (dialog) {
            dialog.remove();
        }
    }
}

// Create global UI manager instance
const uiManager = new UIManager(); 