from distutils.core import setup
from distutils.extension import Extension
from shutil import copyfile

# python setup.py build_ext --inplace
if __name__ == '__main__':
    # files to be included in the extensions
    paths = [
        'Algorithm/FoCore',
        'MLGraph/MLGraph',
        'Utils/Metrics',
        'Utils/Timer',
        'Utils/log',
        'main',
        'constant',
    ]
    # import Cython if available
    USE_CYTHON = True
    try:
        from Cython.Build import cythonize
    except ImportError:
        cythonize = None
        USE_CYTHON = False

    # if Cython is available
    if USE_CYTHON:
        # set the .pyx extension
        ext = '.pyx'

        # create the new .pyx files
        for path in paths:
            copyfile(path + '.py', path + ext)
    # otherwise
    else:
        # set the .c extension
        ext = '.c'

    # build the extensions list
    extensions = [Extension(path.replace('/', '.'), [path + ext]) for path in paths]
    print(extensions)
    # if Cython is available
    if USE_CYTHON:
        # cythonize the extensions
        extensions = cythonize(extensions)

    # run the setup
    setup(
        ext_modules=extensions
    )
