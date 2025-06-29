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

    // Start training
    async startTraining() {
        const fileInput = document.getElementById('fileInput');
        const promptInput = document.getElementById('promptInput');
        
        const file = fileInput.files[0];
        const prompt = promptInput.value.trim();

        if (!file || !prompt) {
            this.showNotification('error', 'Missing Information', 'Please select a file and enter a prompt');
            return;
        }

        try {
            this.showLoading('Starting ML Pipeline', 'Uploading data and parsing prompt...');
            
            const result = await apiClient.startMLSession(file, prompt);
            this.currentRunId = result.run_id;
            
            this.hideLoading();
            
            if (result.status === 'needs_clarification') {
                this.showClarificationDialog(result);
            } else if (result.status === 'training_started') {
                this.handleTrainingStarted(result);
            }
            
        } catch (error) {
            this.hideLoading();
            this.showNotification('error', 'Training Failed', handleError(error, 'Failed to start training'));
        }
    }

    // Handle training started
    handleTrainingStarted(result) {
        this.trainingStartTime = Date.now();
        
        // Update task configuration
        this.updateTaskConfiguration(result.parsed_intent);
        
        // Update data profile
        this.updateDataProfile(result.data_profile);
        
        // Switch to training tab
        this.switchTab('training');
        
        // Update session status
        this.updateSessionStatus('training', 'Training in progress');
        
        // Initialize family progress
        this.initializeFamilyProgress();
        
        // Clear training logs
        this.clearTrainingLogs();
        
        this.showNotification('success', 'Training Started', 'ML pipeline is now running');
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
        logElement.className = 'log-entry';
        
        const timestamp = formatTimestamp(new Date());
        logElement.innerHTML = `
            <span class="log-timestamp">[${timestamp}]</span>
            <span class="log-family">${entry.family}</span>
            Trial ${entry.trial}: <span class="log-metric">${formatNumber(entry.val_metric)}</span>
            (${formatDuration(entry.elapsed_s)})
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
        const clarification = prompt(
            `Clarification needed:\n\n${result.clarifications}\n\nPlease provide clarification:`
        );
        
        if (clarification) {
            this.provideClarification(result.run_id, clarification);
        }
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
}

// Create global UI manager instance
const uiManager = new UIManager(); 