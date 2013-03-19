'''
Created on Mar 13, 2013

@author: Max
'''
import sys, os

root = sys.argv[1]
if not os.path.exists(root):
    sys.exit(1)

if 'jobs' in sys.argv:
    dirstoremove = []
    for parent, folders, files in os.walk(root):
        if "jobs" == os.path.basename(parent):
            dirstoremove.append([parent,files])
            
    for parent,files in dirstoremove:
        for f in files:
            os.remove(os.path.join(parent,f))
        os.rmdir(parent)
        
if 'pdfs' in sys.argv:
    pdfstoremove = []
    for parent, folders, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1] is 'pdf':
                pdfstoremove.append(os.path.join(parent,f))
            
    for pdf in pdfstoremove:
        os.remove(os.path.join(pdf))
    
