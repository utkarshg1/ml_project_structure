from pathlib import Path


def generate_project_structure():
    """
    Generates basic folder structured required for an ML project
    Folders and files will be created only if they do not exists
    """
    print("Starting project scaffolding")

    # Folders
    directories_to_create = ["data", "models", "logs", "src"]

    # Files
    files_to_create = [
        Path("src", "__init__.py"),
        Path("src", "constants.py"),
        Path("src", "logging_config.py"),
        Path("src", "data.py"),
        Path("src", "model_trainer.py"),
        Path("src", "model_evaluator.py"),
        Path("src", "predict.py"),
        Path("data", ".gitkeep"),
        Path("models", ".gitkeep"),
        Path("app.py"),
    ]

    # create directories first
    for directory in directories_to_create:
        path_obj = Path(directory)
        try:
            path_obj.mkdir(parents=True, exist_ok=True)
            print(f"Directory created or already exists : {path_obj}/")
        except Exception as e:
            print(f"Error creating directory : {path_obj} : {e}")

    # create files next
    for file_path in files_to_create:
        try:
            if not file_path.exists():
                file_path.touch()
                print(f"Empty File created : {file_path}")
            else:
                print(f"File already exists, skipping creation for {file_path}")
        except Exception as e:
            print(f"Error creating file {file_path} : {e}")

    print("Project Folder structure generation complete")


def add_logs_to_gitignore():
    """Adds log folder and .log files to .gitignore if not already present."""
    gitignore_path = Path(".gitignore")

    # The content we want to ensure is in .gitignore
    # Note: Added an initial newline for clean appending if file already has content.
    new_ignore_content = "\n# Log folder and files\nlogs/\n*.log\n"

    try:
        # Read existing content, or an empty string if .gitignore doesn't exist
        existing_content = gitignore_path.read_text() if gitignore_path.exists() else ""

        # Check if the primary ignore patterns are already present
        if "logs/" in existing_content and "*.log" in existing_content:
            print("Log ignore entries already present in .gitignore. No changes made.")
            return

        # Append the new content (will create .gitignore if it doesn't exist)
        with open(gitignore_path, "a") as f:
            f.write(new_ignore_content)
        print(f"Log ignore entries added to {gitignore_path}")

    except Exception as e:
        print(f"Error adding logs to .gitignore: {e}")


if __name__ == "__main__":
    generate_project_structure()
    add_logs_to_gitignore()
