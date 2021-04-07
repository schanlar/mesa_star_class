from tempfile import mkstemp
from shutil import move
from os import fdopen, remove
import os, sys


def replace(pattern, subst):

    cwd = os.getcwd()
    file_path = os.path.join(cwd, "mesa_star_class.py")

    # Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh, "w") as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    # Remove original file
    remove(file_path)
    # Move new file
    move(abs_path, file_path)


def main():

    pattern_mesa = f"mesa_dir = {repr('/vol/aibn1107/data2/schanlar/mesa-r10398')}"
    pattern_plots = f"plot_results_dir = {repr('/users/schanlar/Desktop')}"

    subst_mesa_dir = sys.argv[1]
    subst_plots_dir = sys.argv[2]

    subst_mesa = f"mesa_dir = {repr(subst_mesa_dir)}"
    subst_plots = f"plot_results_dir = {repr(subst_plots_dir)}"

    print("Setting paths...")
    replace(pattern_mesa, subst_mesa)
    replace(pattern_plots, subst_plots)
    print("All done!")
    print("*" * 30)


if __name__ == "__main__":
    main()
