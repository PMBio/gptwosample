'''
Created on Mar 13, 2013

@author: Max
'''
import sys, os

root = sys.argv[1]
if not os.path.exists(root):
    sys.exit(1)

dirstoremove = []

for parent, folders, files in os.walk(root):
    if "jobs" == os.path.basename(parent):
        dirstoremove.append(parent)
        
for parent in dirstoremove:
    os.removedirs(parent)
