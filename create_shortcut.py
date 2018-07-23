import sys
import winshell
from os import path
here = path.abspath(__file__)
here = path.dirname(here)
main = path.join(here, 'analysis.py')
link_filepath = path.join(winshell.desktop(), "题目分析.lnk")
with winshell.shortcut(link_filepath) as link:
  link.path = sys.executable
  link.description = "题目分析"
  link.arguments = main
