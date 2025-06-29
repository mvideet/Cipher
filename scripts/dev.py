#!/usr/bin/env python3
"""
Development runner for AutoML Desktop
Starts both backend and frontend in development mode
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


class DevRunner:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True
    
    def cleanup(self, signum=None, frame=None):
        """Cleanup processes on exit"""
        print("\n🛑 Shutting down...")
        self.running = False
        
        if self.backend_process:
            print("Stopping backend...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
        
        if self.frontend_process:
            print("Stopping frontend...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
        
        print("✅ Cleanup complete")
        sys.exit(0)
    
    def check_env(self):
        """Check if .env file exists and has required variables"""
        env_file = Path('.env')
        if not env_file.exists():
            print("❌ .env file not found!")
            print("Run: python scripts/setup.py")
            sys.exit(1)
        
        # Check for OpenAI API key
        with open(env_file) as f:
            content = f.read()
            if 'OPENAI_API_KEY=your_openai_api_key_here' in content:
                print("⚠️  Warning: Please set your OpenAI API key in .env file")
                print("The application will work with limited functionality without it.")
        
        print("✓ Environment file found")
    
    def start_backend(self):
        """Start the Python backend"""
        print("🐍 Starting Python backend...")
        
        try:
            # Try with poetry first
            self.backend_process = subprocess.Popen([
                'poetry', 'run', 'python', '-m', 'uvicorn', 
                'src.main:app', '--reload', '--host', '127.0.0.1', '--port', '8001'
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except FileNotFoundError:
            # Fallback to direct python
            self.backend_process = subprocess.Popen([
                'python', '-m', 'uvicorn', 
                'src.main:app', '--reload', '--host', '127.0.0.1', '--port', '8001'
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        print("✓ Backend starting...")
    
    def wait_for_backend(self, timeout=30):
        """Wait for backend to be ready"""
        import requests
        
        print("⏳ Waiting for backend to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get('http://127.0.0.1:8001/health', timeout=1)
                if response.status_code == 200:
                    print("✅ Backend is ready!")
                    return True
            except requests.exceptions.RequestException:
                time.sleep(1)
        
        print("❌ Backend failed to start within timeout")
        return False
    
    def start_frontend(self):
        """Start the Electron frontend"""
        print("⚡ Starting Electron frontend...")
        
        self.frontend_process = subprocess.Popen([
            'npm', 'start'
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        print("✓ Frontend starting...")
    
    def monitor_processes(self):
        """Monitor both processes"""
        print("\n🔍 Monitoring processes...")
        print("Press Ctrl+C to stop")
        
        while self.running:
            # Check backend
            if self.backend_process.poll() is not None:
                print("❌ Backend process died")
                break
            
            # Check frontend
            if self.frontend_process and self.frontend_process.poll() is not None:
                print("❌ Frontend process died")
                break
            
            time.sleep(1)
    
    def run(self):
        """Main run function"""
        print("🚀 AutoML Desktop Development Server")
        print("=" * 50)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        
        try:
            # Check environment
            self.check_env()
            
            # Start backend
            self.start_backend()
            
            # Wait for backend to be ready
            if not self.wait_for_backend():
                self.cleanup()
                return
            
            # Start frontend
            self.start_frontend()
            
            # Monitor processes
            self.monitor_processes()
            
        except KeyboardInterrupt:
            self.cleanup()
        except Exception as e:
            print(f"❌ Error: {e}")
            self.cleanup()


if __name__ == '__main__':
    runner = DevRunner()
    runner.run() 