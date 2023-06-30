from os import path
import sys
import subprocess

BASE_PATH = path.abspath(path.dirname(__file__))
print(f'Current directory {BASE_PATH}')

def get_requirements(requirements_filename: str):
    requirements_file = path.join(BASE_PATH, requirements_filename)
    with open(requirements_file, 'r', encoding='utf-8') as r:
        requirements = r.readlines()
    requirements = [r.strip() for r in requirements if r.strip()
                    and not r.strip().startswith("#")]
    print(f'Required requirements {requirements}')
    print(f'Installing requirements')
    for req in requirements:
        # implement pip as a subprocess:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
        req])


requirements_filename  = 'requirements.txt'
get_requirements(requirements_filename)



