import glob
import subprocess

#for notebook in glob.glob("*.ipynb"):
for notebook in ['part1.ipynb', 'part2.ipynb']:
    print("make slides for: ", notebook)
    command = "jupyter nbconvert " + notebook + " --to slides --SlidesExporter.reveal_scroll=True"
    subprocess.run(command, shell=True)