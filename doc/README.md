# Pylot Documentation

To compile the documentation, run the following commands from this directory.

```
pip3 install -r requirements-doc.txt
make html
open build/html/index.html
```

To test if there are any build errors with the documentation, do the following.

```
sphinx-build -W -b html -d build/doctrees source build/html
```
