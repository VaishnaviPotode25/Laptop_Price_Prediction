import subprocess

notebooks = [
    "data_ingestion.ipynb",
    "preprocessing_1.ipynb",
    "preprocessing_2.ipynb",
    "preprocessing_3.ipynb",
    "preprocessing_4.ipynb"
]

for nb in notebooks:
    try:
        completed = subprocess.run(
            ["jupyter", "nbconvert", "--execute", "--to", "notebook", "--inplace", nb, "--ExecutePreprocessor.timeout=-1"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"{nb} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {nb}:\n{e.stderr}")
