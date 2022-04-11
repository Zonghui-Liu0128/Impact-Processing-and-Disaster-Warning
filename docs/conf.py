
import sys
import os

# We're working in the ./docs directory, but need the package root in the path
# This command appends the directory one level up, in a cross-platform way.
sys.path.insert(0, os.path.abspath(os.sep.join((os.curdir, '..'))))

project = 'Armageddon'
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.mathjax']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build']
autoclass_content = "both"
pdf_use_index = False
pdf_documents = [('index', 'Armageddon', 'acse4', 'Group Bennu'), ]

