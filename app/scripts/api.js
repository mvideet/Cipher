// API client for AutoML Desktop backend communication

class APIClient {
    constructor() {
        this.baseURL = 'http://127.0.0.1:8001/api/v1';
        this.wsURL = 'ws://127.0.0.1:8001/ws';
        this.sessionId = null;
        this.websocket = null;
        this.listeners = new Map();
        this.reconnectAttempts = 0;
        this.isReconnecting = false;
    }

    // Initialize session
    initSession() {
        this.sessionId = generateSessionId();
        console.log('Session initialized:', this.sessionId);
    }

    // Connect WebSocket for real-time updates
    connectWebSocket() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            return Promise.resolve();
        }
        
        // Prevent multiple connection attempts
        if (this.isConnecting) {
            return this.connectionPromise;
        }

        this.isConnecting = true;

        this.connectionPromise = new Promise((resolve, reject) => {
            const wsUrl = `${this.wsURL}/${this.sessionId}`;
            console.log('Connecting to WebSocket:', wsUrl);

            this.websocket = new WebSocket(wsUrl);
            
            let connectionTimeout;

            this.websocket.onopen = () => {
                clearTimeout(connectionTimeout);
                console.log('WebSocket connected');
                this.isConnecting = false;
                this.reconnectAttempts = 0; // Reset on successful connection
                resolve();
            };

            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };

            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                clearTimeout(connectionTimeout);
                this.websocket = null;
                this.isConnecting = false;
                this.attemptReconnect();
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                clearTimeout(connectionTimeout);
                this.isConnecting = false;
                reject(error);
                this.attemptReconnect();
            };

            // Timeout after 10 seconds
            connectionTimeout = setTimeout(() => {
                if (this.websocket && this.websocket.readyState !== WebSocket.OPEN) {
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
        if (this.isReconnecting || !this.sessionId) return;

        this.isReconnecting = true;
        this.reconnectAttempts++;
        
        // Exponential backoff: 2s, 4s, 8s, 16s, 30s max
        const delay = Math.min(30000, Math.pow(2, this.reconnectAttempts) * 1000);
        
        console.log(`Attempting to reconnect in ${delay / 1000}s...`);

        setTimeout(() => {
            this.isReconnecting = false;
            this.connectWebSocket().catch(err => {
                console.error("Reconnect attempt failed:", err.message);
            });
        }, delay);
    }

    // Handle WebSocket messages
    handleWebSocketMessage(data) {
        console.log('WebSocket message received:', data);
        
        const { type, data: payload } = data;
        
        // Notify all listeners for this message type
        const typeListeners = this.listeners.get(type) || [];
        typeListeners.forEach(callback => {
            try {
                callback(payload);
            } catch (error) {
                console.error('Error in WebSocket listener:', error);
            }
        });

        // Notify global listeners
        const globalListeners = this.listeners.get('*') || [];
        globalListeners.forEach(callback => {
            try {
                callback(type, payload);
            } catch (error) {
                console.error('Error in global WebSocket listener:', error);
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
            const response = await fetch('http://127.0.0.1:8001/health');
            if (response.ok) {
                return await response.json();
            }
            throw new Error('Backend health check failed');
        } catch (error) {
            console.error('Health check failed:', error);
            throw error;
        }
    }

    // Disconnect WebSocket
    disconnect() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        this.sessionId = null;
        this.listeners.clear();
    }

    // Get model recommendations from AI
    async getModelRecommendations(file, prompt, enhanced = false) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('prompt', prompt);
        formData.append('enhanced', enhanced.toString());
        formData.append('session_id', this.sessionId);

        const response = await fetch(`${this.baseURL}/api/get-model-recommendations`, {
            method: 'POST',
            body: formData
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

    // Start training with user-selected models
    async startTrainingWithSelection(selectionData) {
        const response = await fetch(`${this.baseURL}/api/start-training-with-selection`, {
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