// API client for Cipher Desktop backend communication

class APIClient {
    constructor() {
        this.baseURL = 'http://127.0.0.1:8001/api/v1';
        this.wsURL = 'ws://127.0.0.1:8001/ws';
        this.sessionId = null;
        this.websocket = null;
        this.listeners = new Map();
        this.reconnectAttempts = 0;
        this.isReconnecting = false;
        this.connectionEstablished = false;
        this.statusMonitorInterval = null;
        this.keepAliveInterval = null;
        
        // Start connection status monitoring
        this.startConnectionMonitoring();
    }

    // Initialize session
    initSession() {
        this.sessionId = generateSessionId();
        console.log('Session initialized:', this.sessionId);
    }

    // Connect WebSocket for real-time updates
    connectWebSocket() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            console.log('🔗 WebSocket already connected, reusing connection');
            return Promise.resolve();
        }
        
        // Close any existing connection first
        if (this.websocket && this.websocket.readyState !== WebSocket.CLOSED) {
            console.log('🔄 Closing existing WebSocket before new connection');
            try {
                this.websocket.close();
            } catch (error) {
                console.error('❌ Error closing existing WebSocket:', error);
            }
            this.websocket = null;
        }
        
        // Prevent multiple connection attempts
        if (this.isConnecting) {
            console.log('🔗 WebSocket connection already in progress, waiting...');
            return this.connectionPromise;
        }

        this.isConnecting = true;
        console.log('🔗 Starting new WebSocket connection attempt...');

        this.connectionPromise = new Promise((resolve, reject) => {
            const wsUrl = `${this.wsURL}/${this.sessionId}`;
            console.log('🔗 Connecting to WebSocket:', wsUrl);
            console.log('🔗 Session ID:', this.sessionId);
            console.log('🔗 Base WS URL:', this.wsURL);

            this.websocket = new WebSocket(wsUrl);
            
            let connectionTimeout;
            console.log('🔗 WebSocket object created, readyState:', this.websocket.readyState);

            this.websocket.onopen = (event) => {
                clearTimeout(connectionTimeout);
                console.log('✅ WebSocket connected successfully!');
                console.log('🔗 Connection event:', event);
                console.log('🔗 WebSocket readyState:', this.websocket.readyState);
                this.isConnecting = false;
                this.reconnectAttempts = 0; // Reset on successful connection
                
                // Small delay to ensure connection is fully stabilized
                setTimeout(() => {
                    this.connectionEstablished = true;
                    console.log('🎉 WebSocket connection fully established and ready');
                    
                    // Start keep-alive ping every 20 seconds
                    this.startKeepAlive();
                    
                    resolve();
                }, 100);
            };

            this.websocket.onmessage = (event) => {
                console.log('📨 WebSocket message received:', event.data);
                try {
                    const data = JSON.parse(event.data);
                    console.log('📨 Parsed message:', data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('❌ Failed to parse WebSocket message:', error);
                    console.error('❌ Raw message:', event.data);
                }
            };

            this.websocket.onclose = (event) => {
                console.log('🔌 WebSocket disconnected');
                console.log('🔌 Close event details:', {
                    code: event.code,
                    reason: event.reason,
                    wasClean: event.wasClean
                });
                clearTimeout(connectionTimeout);
                this.stopKeepAlive(); // Stop keep-alive on disconnect
                this.websocket = null;
                this.isConnecting = false;
                this.connectionEstablished = false;
                this.attemptReconnect();
            };

            this.websocket.onerror = (error) => {
                console.error('❌ WebSocket error occurred');
                console.error('❌ Error event:', error);
                console.error('❌ WebSocket readyState:', this.websocket ? this.websocket.readyState : 'null');
                clearTimeout(connectionTimeout);
                this.isConnecting = false;
                reject(error);
                this.attemptReconnect();
            };

            // Timeout after 10 seconds
            connectionTimeout = setTimeout(() => {
                console.error('⏰ WebSocket connection timeout after 10 seconds');
                console.error('⏰ WebSocket readyState:', this.websocket ? this.websocket.readyState : 'null');
                if (this.websocket && this.websocket.readyState !== WebSocket.OPEN) {
                    console.log('⏰ Closing timed-out connection');
                    this.websocket.close(); // Close the attempting connection
                    this.isConnecting = false;
                    reject(new Error('WebSocket connection timeout'));
                    this.attemptReconnect();
                }
            }, 10000);
        });

        return this.connectionPromise;
    }

    // Attempt to reconnect with exponential backoff
    attemptReconnect() {
        if (this.isReconnecting || !this.sessionId) {
            console.log('🔄 Reconnect skipped: already reconnecting or no session');
            return;
        }

        this.isReconnecting = true;
        this.reconnectAttempts++;
        
        // Exponential backoff: 2s, 4s, 8s, 16s, 30s max
        const delay = Math.min(30000, Math.pow(2, this.reconnectAttempts) * 1000);
        
        console.log(`🔄 Attempting to reconnect in ${delay / 1000}s... (attempt ${this.reconnectAttempts})`);
        console.log('🔄 Session ID for reconnect:', this.sessionId);

        // Check backend health before reconnecting
        this.checkBackendHealth().then(isHealthy => {
            console.log('🏥 Backend health check result:', isHealthy);
            
            setTimeout(() => {
                this.isReconnecting = false;
                console.log('🔄 Starting reconnection attempt...');
                this.connectWebSocket().catch(err => {
                    console.error("❌ Reconnect attempt failed:", err.message);
                    console.error("❌ Full error:", err);
                });
            }, delay);
        }).catch(healthErr => {
            console.error('🏥 Backend health check failed:', healthErr);
            // Still try to reconnect even if health check fails
            setTimeout(() => {
                this.isReconnecting = false;
                console.log('🔄 Starting reconnection attempt (despite health check failure)...');
                this.connectWebSocket().catch(err => {
                    console.error("❌ Reconnect attempt failed:", err.message);
                });
            }, delay);
        });
    }

    // Handle WebSocket messages
    handleWebSocketMessage(data) {
        console.log('📨 WebSocket message received:', data);
        
        const { type, data: payload } = data;
        
        // Handle special system messages
        if (type === 'connection_established') {
            console.log('🎉 WebSocket connection establishment confirmed by server');
            this.connectionEstablished = true;
        } else if (type === 'keepalive') {
            console.log('💓 Keepalive received from server');
        } else if (type === 'pong') {
            console.log('🏓 Pong received from server');
        }
        
        // Notify all listeners for this message type
        const typeListeners = this.listeners.get(type) || [];
        typeListeners.forEach(callback => {
            try {
                callback(payload);
            } catch (error) {
                console.error('❌ Error in WebSocket listener:', error, 'for type:', type);
            }
        });

        // Notify global listeners
        const globalListeners = this.listeners.get('*') || [];
        globalListeners.forEach(callback => {
            try {
                callback(type, payload);
            } catch (error) {
                console.error('❌ Error in global WebSocket listener:', error, 'for type:', type);
            }
        });
    }

    // Add WebSocket event listener
    addEventListener(type, callback) {
        if (!this.listeners.has(type)) {
            this.listeners.set(type, []);
        }
        this.listeners.get(type).push(callback);
    }

    // Remove WebSocket event listener
    removeEventListener(type, callback) {
        const typeListeners = this.listeners.get(type);
        if (typeListeners) {
            const index = typeListeners.indexOf(callback);
            if (index > -1) {
                typeListeners.splice(index, 1);
            }
        }
    }

    // Generic HTTP request
    async request(method, endpoint, data = null, isFormData = false) {
        const url = `${this.baseURL}${endpoint}`;
        const headers = {};

        if (!isFormData) {
            headers['Content-Type'] = 'application/json';
        }

        const config = {
            method,
            headers,
        };

        if (data) {
            if (isFormData) {
                config.body = data;
            } else {
                config.body = JSON.stringify(data);
            }
        }

        try {
            console.log(`Making ${method} request to ${url}`);
            const response = await fetch(url, config);

            if (!response.ok) {
                let errorMessage;
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.detail || errorData.message || errorData.error;
                } catch (e) {
                    errorMessage = `HTTP ${response.status}: ${response.statusText}`;
                }
                const error = new Error(errorMessage || `HTTP ${response.status}: ${response.statusText}`);
                error.status = response.status;
                error.statusText = response.statusText;
                throw error;
            }

            const result = await response.json();
            console.log(`${method} response:`, result);
            return result;

        } catch (error) {
            console.error(`${method} request failed:`, error);
            throw error;
        }
    }

    // Start ML session with file upload and prompt
    async startMLSession(file, prompt, enhanced = false) {
        if (!this.sessionId) {
            throw new Error('Session not initialized');
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('prompt', prompt);

        const endpoint = enhanced ? 'start-enhanced' : 'start';
        return this.request(
            'POST',
            `/session/${this.sessionId}/${endpoint}`,
            formData,
            true
        );
    }

    // Provide clarification
    async provideClarification(runId, clarification) {
        if (!this.sessionId) {
            throw new Error('Session not initialized');
        }

        const formData = new FormData();
        formData.append('run_id', runId);
        formData.append('clarification', clarification);

        return this.request(
            'POST',
            `/session/${this.sessionId}/clarify`,
            formData,
            true
        );
    }

    // Get run status
    async getRunStatus(runId) {
        if (!this.sessionId) {
            throw new Error('Session not initialized');
        }

        return this.request(
            'GET',
            `/session/${this.sessionId}/status/${runId}`
        );
    }

    // Deploy model
    async deployModel(runId) {
        if (!this.sessionId) {
            throw new Error('Session not initialized');
        }

        return this.request(
            'POST',
            `/session/${this.sessionId}/deploy/${runId}`
        );
    }

    // Check backend health
    async checkHealth() {
        try {
            console.log('🏥 Checking backend health at http://127.0.0.1:8001/api/v1/health');
            const response = await fetch(`${this.baseURL}/health`);
            console.log('🏥 Health check response status:', response.status);
            console.log('🏥 Health check response ok:', response.ok);
            
            if (response.ok) {
                const healthData = await response.json();
                console.log('🏥 Backend health check passed:', healthData);
                return healthData;
            }
            throw new Error(`Backend health check failed with status ${response.status}`);
        } catch (error) {
            console.error('🏥 Health check failed:', error);
            throw error;
        }
    }

    // Check backend health returning boolean
    async checkBackendHealth() {
        try {
            await this.checkHealth();
            return true;
        } catch (error) {
            console.error('🏥 Backend not healthy:', error);
            return false;
        }
    }

    // Start connection monitoring
    startConnectionMonitoring() {
        // Log connection status every 10 seconds
        this.statusMonitorInterval = setInterval(() => {
            this.logConnectionStatus();
        }, 10000);
    }

    // Start keep-alive mechanism
    startKeepAlive() {
        // Clear any existing interval
        this.stopKeepAlive();
        
        console.log('💓 Starting WebSocket keep-alive mechanism');
        
        // Send ping every 20 seconds
        this.keepAliveInterval = setInterval(() => {
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                try {
                    this.websocket.send('ping');
                    console.log('💓 Keep-alive ping sent');
                } catch (error) {
                    console.error('❌ Failed to send keep-alive ping:', error);
                    this.stopKeepAlive();
                }
            } else {
                console.log('⚠️ WebSocket not open, stopping keep-alive');
                this.stopKeepAlive();
            }
        }, 20000); // Every 20 seconds
    }

    // Stop keep-alive mechanism
    stopKeepAlive() {
        if (this.keepAliveInterval) {
            clearInterval(this.keepAliveInterval);
            this.keepAliveInterval = null;
            console.log('💔 Keep-alive mechanism stopped');
        }
    }

    // Log current connection status
    logConnectionStatus() {
        const status = {
            sessionId: this.sessionId,
            hasWebSocket: !!this.websocket,
            readyState: this.websocket ? this.websocket.readyState : 'No WebSocket',
            readyStateText: this.websocket ? this.getReadyStateText(this.websocket.readyState) : 'No WebSocket',
            isConnecting: this.isConnecting,
            isReconnecting: this.isReconnecting,
            reconnectAttempts: this.reconnectAttempts,
            connectionEstablished: this.connectionEstablished
        };

        console.log('📊 WebSocket Status:', status);

        // Additional health checks
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            // Send ping to test connection
            try {
                this.websocket.send('ping');
                console.log('🏓 Ping sent to server');
            } catch (error) {
                console.error('❌ Failed to send ping:', error);
            }
        }
    }

    // Get human-readable ready state
    getReadyStateText(readyState) {
        switch (readyState) {
            case WebSocket.CONNECTING: return 'CONNECTING (0)';
            case WebSocket.OPEN: return 'OPEN (1)';
            case WebSocket.CLOSING: return 'CLOSING (2)';
            case WebSocket.CLOSED: return 'CLOSED (3)';
            default: return `UNKNOWN (${readyState})`;
        }
    }

    // Disconnect WebSocket
    disconnect() {
        console.log('🔌 Disconnecting WebSocket and cleaning up...');
        
        // Stop monitoring
        if (this.statusMonitorInterval) {
            clearInterval(this.statusMonitorInterval);
            this.statusMonitorInterval = null;
        }
        
        // Stop keep-alive
        this.stopKeepAlive();
        
        if (this.websocket) {
            console.log('🔌 Closing WebSocket connection...');
            this.websocket.close();
            this.websocket = null;
        }
        this.sessionId = null;
        this.listeners.clear();
        this.connectionEstablished = false;
        console.log('✅ WebSocket disconnected and cleaned up');
    }

    // Get model recommendations (for enhanced mode)
    async getModelRecommendations(file, prompt, enhanced) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('prompt', prompt);
        formData.append('enhanced', enhanced);
        formData.append('session_id', this.sessionId);

        const response = await fetch(`${this.baseURL}/get-model-recommendations`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    // Generate query suggestions based on uploaded dataset
    async generateQuerySuggestions(file, maxSuggestions = 5) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('max_suggestions', maxSuggestions);

        const response = await fetch(`${this.baseURL}/generate-query-suggestions`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    // Analyze dataset with optional suggestions
    async analyzeDataset(file, includeSuggestions = true) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('include_suggestions', includeSuggestions);

        const response = await fetch(`${this.baseURL}/analyze-dataset`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    // Start training with user-selected models
    async startTrainingWithSelection(selectionData) {
        const response = await fetch(`${this.baseURL}/start-training-with-selection`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(selectionData)
        });

        if (!response.ok) {
            let errorMessage;
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorData.message || errorData.error;
            } catch (e) {
                errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            }
            const error = new Error(errorMessage || `HTTP ${response.status}: ${response.statusText}`);
            error.status = response.status;
            error.statusText = response.statusText;
            throw error;
        }

        return response.json();
    }
}

// Create global API client instance
const apiClient = new APIClient(); 