#!/usr/bin/env python3
"""
Setup script for Cipher Desktop
Automates the installation process
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run a command and handle errors"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, check=check, 
            capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e.stderr}")
        sys.exit(1)


def check_requirements():
    """Check if required software is installed"""
    requirements = {
        'python': ['python', '--version'],
        'node': ['node', '--version'],
        'npm': ['npm', '--version'],
        'docker': ['docker', '--version']
    }
    
    missing = []
    for name, cmd in requirements.items():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✓ {name}: {version}")
            else:
                missing.append(name)
        except FileNotFoundError:
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing requirements: {', '.join(missing)}")
        print("\nPlease install the missing software and try again.")
        sys.exit(1)
    
    print("\n✓ All requirements satisfied")


def setup_environment():
    """Set up environment file"""
    env_template = Path('.env.template')
    env_file = Path('.env')
    
    if env_file.exists():
        print("✓ .env file already exists")
        return
    
    if env_template.exists():
        import shutil
        shutil.copy(env_template, env_file)
        print("✓ Created .env file from template")
        print("\n⚠️  IMPORTANT: Edit .env file and add your OpenAI API key!")
    else:
        print("❌ .env.template not found")


def install_python_deps():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    # Check if poetry is installed
    try:
        subprocess.run(['poetry', '--version'], capture_output=True, check=True)
        print("✓ Poetry found")
        run_command('poetry install')
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Poetry not found, installing...")
        run_command('pip install poetry')
        run_command('poetry install')


def install_node_deps():
    """Install Node.js dependencies"""
    print("\n📦 Installing Node.js dependencies...")
    run_command('npm install')


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    directories = ['temp', 'models', 'runs']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created {dir_name}/")


def main():
    """Main setup function"""
    print("🚀 Cipher Desktop Setup")
    print("=" * 50)
    
    # Check requirements
    print("\n1. Checking requirements...")
    check_requirements()
    
    # Setup environment
    print("\n2. Setting up environment...")
    setup_environment()
    
    # Create directories
    print("\n3. Creating directories...")
    create_directories()
    
    # Install Python dependencies
    print("\n4. Installing Python dependencies...")
    install_python_deps()
    
    # Install Node.js dependencies
    print("\n5. Installing Node.js dependencies...")
    install_node_deps()
    
    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Run: npm run dev")
    print("\nFor production build: npm run build")


if __name__ == '__main__':
    main() 