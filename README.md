# FocusCore Decomposition of Multilayer Graphs

## Dataset

Download datasets and put them into the /Dataset folder.

Each multilayer graph should be stored in a .txt file, and the filename is considered the dataset name.

The first line of the .txt file should contain ``#layers, #vertex, #edges``.

Following the header, each row contains three integers, which are ``layer-id, node-id, node-id``.

An example file "homo" is provided, and new datasets should adhere to the same structure.

We also include a script to convert ``WikiTalk`` and ``StackOverflow`` from original snap format ([Wiki](https://snap.stanford.edu/data/wiki-Talk.html), [SO](https://snap.stanford.edu/data/sx-stackoverflow.html)) to multilayer graph. Refer to "ConvertWiki.py" and "CovertSO.py" for usage.

The two case study datasets are also included as ``dblp-ijcai-kdd-mod.txt`` and ``dblp-ijcai-kdd-mod-aaai.txt``. ``metainfo_id_author.txt`` shows a map from ``node-id`` to ``author name`` for ``dblp-ijcai-kdd-mod.txt``. For more information about how to convert DBLP dataset to multilayer graphs, check our previous code at [here](https://github.com/MDCGraph/DBLP-MLG).

The missing datasets are available at [FirmCore](https://github.com/joint-em/FTCS/tree/main/Code/Datasets) and [multilayer kCore](https://github.com/egalimberti/multilayer_core_decomposition).

## Usage

### Accelerate by Cython

Our code provides an acceleration using Cython as follows:

```shell
python setup.py build_ext --inplace
```

Without using Cython, the codes also works.

And you can use "clean.py" to clean the files created by Cython.

In general, the code can be run as follows:

```shell
python main.py -a [Decomposition/Subgraph] -d [dataset name(default homo)] -m [EP/IP/VC(default)]
```

### Decomposition

```shell
python main.py -a Decomposition -d [dataset(default homo)] -m [EP/IP/VC(default)]
```

examples:

Using Vertex Centric to compute FoCore Decomposition of Homo Dataset:

```shell
python main.py -a Decomposition -d homo -m VC
```

Using Interleaved Peeling to compute FoCore Decomposition of Amazon Dataset:

```shell
python main.py -a Decomposition -d Amazon -m IP
```

### Denest Subgraph

```shell
python main.py -a Subgraph -d [dataset(default homo)]
```

examples:

Compute denest subgraph of Homo Dataset:

```shell
python main.py -a Subgraph -d homo
```

### FoCore Cache

```shell
python main.py -a Cache -d [dataset(default homo)]
```

examples:

Compute denest subgraph of Homo Dataset:

```shell
python main.py -a Cache -d homo
```

## To Cite This Paper

```bibtex
@inproceedings{wangFocusCoreDecompositionMultilayer2024,
  title = {{{FocusCore Decomposition}} of {{Multilayer Graphs}}},
  booktitle = {2024 {{IEEE}} 40th {{International Conference}} on {{Data Engineering}} ({{ICDE}})},
  author = {Wang, Run-An and Liu, Dandan and Zou, Zhaonian},
  year = {2024},
  pages = {2792--2804},
  issn = {2375-026X},
  doi = {10.1109/ICDE60146.2024.00218},
  url = {https://ieeexplore.ieee.org/document/10597766},
}

```
