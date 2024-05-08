try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'icg_benchmark.third_party.libmcubes.mcubes',
    sources=[
        'icg_benchmark/third_party/libmcubes/mcubes.pyx',
        'icg_benchmark/third_party/libmcubes/pywrapper.cpp',
        'icg_benchmark/third_party/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'icg_benchmark.third_party.libmesh.triangle_hash',
    sources=[
        'icg_benchmark/third_party/libmesh/triangle_hash.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir]
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'icg_benchmark.third_party.libmise.mise',
    sources=[
        'icg_benchmark/third_party/libmise/mise.pyx'
    ],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'icg_benchmark.third_party.libsimplify.simplify_mesh',
    sources=[
        'icg_benchmark/third_party/libsimplify/simplify_mesh.pyx'
    ],
    include_dirs=[numpy_include_dir]
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'icg_benchmark.third_party.libvoxelize.voxelize',
    sources=[
        'icg_benchmark/third_party/libvoxelize/voxelize.pyx'
    ],
    libraries=['m']  # Unix-like specific
)

# Gather all extension modules
ext_modules = [
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
]

setup(
    name = "icg_benchmark",
    version = "0.1",
    packages = find_packages(),
    include_package_data=True,
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)
