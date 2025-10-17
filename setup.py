import os
import subprocess
import sys

VENV_DIR = ".venv"

def run_command(command, shell=False):
    """Runs a command and exits if it fails."""
    print(f"🚀 Running: {' '.join(command)}")
    result = subprocess.run(command, shell=shell)
    if result.returncode != 0:
        print(f"🚨 Error: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    """Sets up the project environment."""
    print("--- Starting Project Setup ---")

    # 1. Create virtual environment if it doesn't exist
    if not os.path.isdir(VENV_DIR):
        print(f"🐍 Creating virtual environment at '{VENV_DIR}'...")
        run_command([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print("✅ Virtual environment already exists.")

    # Determine the path to the python/pip executables in the venv
    if sys.platform == "win32":
        pip_executable = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        pip_executable = os.path.join(VENV_DIR, "bin", "pip")

    # 2. Install dependencies
    print("📦 Installing dependencies from requirements.txt...")
    run_command([pip_executable, "install", "-r", "requirements.txt"])

    # 3. Install pre-commit hooks
    pre_commit_executable = pip_executable.replace("pip", "pre-commit")
    print("⚙️ Installing pre-commit hooks...")
    run_command([pre_commit_executable, "install"])

    print("\n🎉 Setup complete! To activate the virtual environment, run:")
    if sys.platform == "win32":
        print(f"   {VENV_DIR}\\Scripts\\activate")
    else:
        print(f"   source {VENV_DIR}/bin/activate")

if __name__ == "__main__":
    main()