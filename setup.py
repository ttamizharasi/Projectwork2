"""
FractureSense AI - Automated Setup and Diagnostic Script
Checks system requirements, installs dependencies, and diagnoses issues
"""

import sys
import os
import subprocess
import platform


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_success(text):
    """Print success message"""
    print(f"✓ {text}")


def print_error(text):
    """Print error message"""
    print(f"✗ {text}")


def print_warning(text):
    """Print warning message"""
    print(f"⚠ {text}")


def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Python version: {version_str}")
    print(f"Platform: {platform.system()}")
    print(f"Architecture: {platform.machine()}")
    
    if version.major == 3 and version.minor >= 8:
        print_success("Python version is compatible (3.8+)")
        return True
    else:
        print_error("Python 3.8 or higher required")
        print("Please install Python from: https://www.python.org/downloads/")
        return False


def check_pip():
    """Check if pip is installed"""
    print_header("Checking pip Installation")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(result.stdout.strip())
            print_success("pip is installed")
            return True
        else:
            print_error("pip is not working properly")
            return False
            
    except Exception as e:
        print_error(f"pip check failed: {e}")
        return False


def check_virtual_environment():
    """Check if running in virtual environment"""
    print_header("Checking Virtual Environment")
    
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        print_success("Running in virtual environment")
        print(f"Virtual environment path: {sys.prefix}")
    else:
        print_warning("Not running in virtual environment")
        print("Recommendation: Create and activate a virtual environment")
        print("\nWindows:")
        print("  python -m venv venv")
        print("  venv\\Scripts\\activate")
        print("\nMac/Linux:")
        print("  python3 -m venv venv")
        print("  source venv/bin/activate")
    
    return in_venv


def install_dependencies():
    """Install required Python packages"""
    print_header("Installing Dependencies")
    
    try:
        print("Installing packages from requirements.txt...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print_success("All dependencies installed successfully")
            print("\nInstalled packages:")
            for line in result.stdout.split('\n'):
                if 'Successfully installed' in line:
                    print(f"  {line.strip()}")
            return True
        else:
            print_error("Installation failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print_error(f"Installation error: {e}")
        return False


def check_installed_packages():
    """Verify all required packages are installed"""
    print_header("Verifying Installed Packages")
    
    required_packages = {
        'Flask': 'flask',
        'Werkzeug': 'werkzeug',
        'Pillow': 'PIL',
        'NumPy': 'numpy'
    }
    
    all_installed = True
    
    for display_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print_success(f"{display_name} is installed")
        except ImportError:
            print_error(f"{display_name} is NOT installed")
            all_installed = False
    
    return all_installed


def check_project_structure():
    """Verify project files and folders exist"""
    print_header("Checking Project Structure")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'utils/predict.py',
        'templates/index.html',
        'static/style.css'
    ]
    
    required_dirs = [
        'utils',
        'templates',
        'static',
        'uploads'
    ]
    
    all_exist = True
    
    print("Files:")
    for file in required_files:
        if os.path.exists(file):
            print_success(f"{file}")
        else:
            print_error(f"{file} - MISSING")
            all_exist = False
    
    print("\nDirectories:")
    for directory in required_dirs:
        if os.path.exists(directory):
            print_success(f"{directory}/")
        else:
            print_error(f"{directory}/ - MISSING")
            all_exist = False
    
    return all_exist


def check_port_availability():
    """Check if default port 5000 is available"""
    print_header("Checking Port Availability")
    
    import socket
    
    port = 5000
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        sock.bind(('127.0.0.1', port))
        sock.close()
        print_success(f"Port {port} is available")
        return True
    except socket.error:
        print_warning(f"Port {port} is already in use")
        print("You can change the port in app.py:")
        print("  app.run(port=5001)")
        return False


def run_diagnostics():
    """Run all diagnostic checks"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  FractureSense AI - System Diagnostic Tool".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    
    checks = [
        ("Python Version", check_python_version),
        ("pip Installation", check_pip),
        ("Virtual Environment", check_virtual_environment),
        ("Project Structure", check_project_structure),
        ("Port Availability", check_port_availability)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print_error(f"{check_name} check failed: {e}")
            results[check_name] = False
    
    # Summary
    print_header("Diagnostic Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name:.<40} {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print_success("All checks passed! System is ready.")
        return True
    else:
        print_warning("Some checks failed. Review the errors above.")
        return False


def setup_and_install():
    """Complete setup process"""
    print_header("Starting Automated Setup")
    
    # Run diagnostics first
    if not run_diagnostics():
        print("\n⚠ Fix the issues above before proceeding with installation.")
        
        response = input("\nAttempt installation anyway? (y/n): ").lower()
        if response != 'y':
            print("Setup cancelled.")
            return False
    
    # Install dependencies
    if not install_dependencies():
        print_error("Dependency installation failed")
        return False
    
    # Verify installation
    if not check_installed_packages():
        print_error("Package verification failed")
        return False
    
    # Final summary
    print_header("Setup Complete!")
    print("\n✓ All dependencies installed")
    print("✓ Project structure verified")
    print("\nNext steps:")
    print("1. Run the application:")
    print("   python app.py")
    print("\n2. Open your browser:")
    print("   http://127.0.0.1:5000")
    print("\n3. Upload an X-ray image to test")
    
    return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FractureSense AI Setup and Diagnostic Tool"
    )
    parser.add_argument(
        '--install',
        action='store_true',
        help='Install dependencies after diagnostics'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only run diagnostics, skip installation'
    )
    
    args = parser.parse_args()
    
    if args.check_only:
        run_diagnostics()
    elif args.install:
        setup_and_install()
    else:
        # Interactive mode
        print("\nFractureSense AI Setup Tool")
        print("1. Run diagnostics only")
        print("2. Install dependencies")
        print("3. Full setup (diagnostics + install)")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            run_diagnostics()
        elif choice == '2':
            install_dependencies()
            check_installed_packages()
        elif choice == '3':
            setup_and_install()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
