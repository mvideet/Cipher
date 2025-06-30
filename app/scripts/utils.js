// Utility functions for AutoML Desktop

// Generate unique session ID
function generateSessionId() {
    return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Format number with appropriate precision
function formatNumber(num, precision = 3) {
    if (typeof num !== 'number') return num;
    if (Math.abs(num) < 0.001) return num.toExponential(2);
    return Number(num.toFixed(precision));
}

// Format duration in seconds to human readable
function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
}

// Format timestamp
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Deep clone object
function deepClone(obj) {
    return JSON.parse(JSON.stringify(obj));
}

// Validate CSV file
function validateCSVFile(file) {
    const errors = [];
    
    if (!file) {
        errors.push('No file selected');
        return errors;
    }
    
    // Check file type
    const allowedTypes = ['text/csv', 'application/csv'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(file.type) && fileExtension !== 'csv') {
        errors.push('File must be a CSV file');
    }
    
    // Check file size (100MB limit)
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
        errors.push(`File size exceeds ${formatFileSize(maxSize)} limit`);
    }
    
    // Check if file is empty
    if (file.size === 0) {
        errors.push('File is empty');
    }
    
    return errors;
}

// Parse CSV data preview
function parseCSVPreview(csvText, maxRows = 10) {
    const lines = csvText.split('\n').filter(line => line.trim());
    if (lines.length === 0) return null;
    
    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
    const rows = [];
    
    for (let i = 1; i < Math.min(lines.length, maxRows + 1); i++) {
        const row = lines[i].split(',').map(cell => cell.trim().replace(/"/g, ''));
        if (row.length === headers.length) {
            const rowObj = {};
            headers.forEach((header, index) => {
                rowObj[header] = row[index];
            });
            rows.push(rowObj);
        }
    }
    
    return {
        headers,
        rows,
        totalRows: lines.length - 1,
        totalColumns: headers.length
    };
}

// Create HTML table from data
function createTable(data) {
    if (!data || !data.headers || !data.rows) return '';
    
    let html = '<table class="data-table">';
    
    // Headers
    html += '<thead><tr>';
    data.headers.forEach(header => {
        html += `<th>${escapeHtml(header)}</th>`;
    });
    html += '</tr></thead>';
    
    // Rows
    html += '<tbody>';
    data.rows.forEach(row => {
        html += '<tr>';
        data.headers.forEach(header => {
            const value = row[header] || '';
            html += `<td>${escapeHtml(value)}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody>';
    
    html += '</table>';
    return html;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Copy text to clipboard
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch (err) {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        const success = document.execCommand('copy');
        document.body.removeChild(textArea);
        return success;
    }
}

// Color palette for charts
const CHART_COLORS = [
    '#3b82f6', '#ef4444', '#10b981', '#f59e0b',
    '#8b5cf6', '#06b6d4', '#84cc16', '#f97316',
    '#ec4899', '#6b7280', '#14b8a6', '#eab308'
];

// Get color by index
function getChartColor(index) {
    return CHART_COLORS[index % CHART_COLORS.length];
}

// Format percentage
function formatPercentage(value, decimals = 1) {
    return (value * 100).toFixed(decimals) + '%';
}

// Check if value is numeric
function isNumeric(value) {
    return !isNaN(parseFloat(value)) && isFinite(value);
}

// Sanitize filename for download
function sanitizeFilename(filename) {
    return filename.replace(/[^a-z0-9.-]/gi, '_').toLowerCase();
}

// Local storage helpers
const storage = {
    set(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (e) {
            console.error('Failed to save to localStorage:', e);
        }
    },
    
    get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            console.error('Failed to read from localStorage:', e);
            return defaultValue;
        }
    },
    
    remove(key) {
        try {
            localStorage.removeItem(key);
        } catch (e) {
            console.error('Failed to remove from localStorage:', e);
        }
    }
};

// Error handling
function handleError(error, context = '') {
    console.error('Error:', error, context);
    
    let message = 'An unexpected error occurred';
    
    if (error && typeof error === 'object') {
        // Handle Response objects from fetch
        if (error.status && error.statusText) {
            message = `HTTP ${error.status}: ${error.statusText}`;
        }
        // Handle error objects with detail property (FastAPI format)
        else if (error.detail) {
            message = error.detail;
        }
        // Handle error objects with message property
        else if (error.message) {
            message = error.message;
        }
        // Handle error objects with error property
        else if (error.error) {
            message = error.error;
        }
        // Handle objects by converting to string
        else {
            try {
                message = JSON.stringify(error);
            } catch (e) {
                message = String(error);
            }
        }
    } else if (typeof error === 'string') {
        message = error;
    } else if (error) {
        message = String(error);
    }
    
    if (context) {
        message = `${context}: ${message}`;
    }
    
    return message;
} 