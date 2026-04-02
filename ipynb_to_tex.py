import nbformat
import sys

def extraire_cellule(fichier_ipynb, num_cellule):
    with open(fichier_ipynb, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    if num_cellule < 0 or num_cellule >= len(notebook.cells):
        print(f"Erreur : La cellule {num_cellule} n'existe pas.")
        return ""

    cellule = notebook.cells[num_cellule]
    if cellule.cell_type == "code":
        return cellule.source
    else:
        print(f"La cellule {num_cellule} n'est pas une cellule de code.")
        return ""

if __name__ == "__main__":
    fichier = sys.argv[1]
    num = int(sys.argv[2])
    code = extraire_cellule(fichier, num)
    print(code)
