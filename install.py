from subprocess import call
from os import path

here = path.abspath(__file__)
here = path.dirname(here)
requ = path.join(here, 'requirements.txt')
call(["pip", "install", "-r", requ])


