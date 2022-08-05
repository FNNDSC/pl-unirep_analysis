pl-unirep_analysis
================================

.. image:: https://img.shields.io/docker/v/fnndsc/pl-unirep_analysis?sort=semver
    :target: https://hub.docker.com/r/fnndsc/pl-unirep_analysis

.. image:: https://img.shields.io/github/license/fnndsc/pl-unirep_analysis
    :target: https://github.com/FNNDSC/pl-unirep_analysis/blob/master/LICENSE

.. image:: https://github.com/FNNDSC/pl-unirep_analysis/workflows/ci/badge.svg
    :target: https://github.com/FNNDSC/pl-unirep_analysis/actions

.. contents:: Table of Contents


Abstract
--------

``unirep_analysis`` is a ChRIS app that is wrapped around the UniRep project (https://github.com/churchlab/UniRep)

This plugin is GPU-capable. The 64-unit model should be OK to run on any machine. The full-sized model will require a machine with more than 8GB of GPU RAM.


Citations
---------

For full information about the underlying method, consult the UniRep publication:

            Paper: https://www.nature.com/articles/s41592-019-0598-1


The source code of UniRep is available on Github: https://github.com/churchlab/UniRep.


Synopsis
--------

.. code::

        unirep_analysis                                                     \
                                    [--dimension <modelDimension>]          \
                                    [--batch_size <batchSize>]              \
                                    [--learning_rate <learningRate>]        \
                                    [--inputFile <inputFileToProcess>]      \
                                    [--inputGlob <inputGlobPattern>]        \
                                    [--modelWeightPath <pathToWeights>]     \
                                    [--outputFile <resultOutputFile>]       \
                                    [--json]                                \
                                    <inputDir>
                                    <outputDir>

Description
-----------

``unirep_analysis`` is a ChRIS-based "plugin" application that is capable of inferencing protein sequence representations and generative modelling aka "babbling".

TL;DR
------

Simply pull the docker image,

.. code::

    docker pull fnndsc/pl-unirep_analysis

and go straight to the examples section.

Arguments
---------

.. code::

        [--dimension <modelDimension>]
        By default, the <modelDimension> is 64. However, the value can be changed
        to 1900 (full) or 256 and the corresponding weights files (present inside
        the container) will be used.

        [--batch_size <batchSize>]
        This represents the batch size of the babbler. Default value is 12.

        [--learning_rate <learningRate>]
        Needed to build the model. Default is 0.001.

        [--inputFile <inputFileToProcess>]
        The name of the input text file that contains your amino acid sequences.
        The default file name is an empty string. The full path to the
        <inputFileToProcess> is constructed by concatenating <inputDir>

                <inputDir>/<inputFileToProcess>

        [--inputGlob <inputGlob>]
        A glob pattern string, default '**/*txt', that specifies the file containing
        an amino acid sequence. This parameter allows for dynamic searching in the
        input space a sequence file, and the first "hit" is grabbed.

        [--modelWeightPath <path>]
        A path to a directory containing model weight files to use for inference.

        [--outputFile <resultOutputFile>]
        The name of the output or formatted 'txt' file. Default name is 'format.txt'

        [-h]
        Display inline help

        [--json]
        If specified, print a JSON representation of the app.

Run
----

The execute vector of this plugin is via ``docker``.

Using ``docker run``
~~~~~~~~~~~~~~~~~~~~

To run using ``docker``, be sure to assign an "input" directory to ``/incoming`` and an output directory to ``/outgoing``. *Make sure that the* ``$(pwd)/out`` *directory is world writable!*

Now, prefix all calls with

.. code:: bash

    docker run --rm -v $(pwd)/out:/outgoing                        \
            fnndsc/pl-unirep_analysis                              \
            unirep_analysis                                        \

Thus, getting inline help is:

.. code:: bash

    mkdir in out && chmod 777 out
    docker run --rm -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing      \
            fnndsc/pl-unirep_analysis                                   \
            unirep_analysis                                             \
            -h                                                          \
            /incoming /outgoing

Examples
--------

Assuming that the ``<inputDir>`` layout conforms to

.. code:: bash

    <inputDir>
        │
        └──█ sequence.txt


to process this (by default on a GPU) do

.. code:: bash

   docker run   --rm --gpus all                                             \
                -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing              \
                fnndsc/pl-unirep_analysis unirep_analysis                   \
                --inputFile sequence.txt --outputFile formatted.txt         \
                /incoming /outgoing

(note the ``--gpus all`` is not necessarily required) which will create in the ``<outputDir>``:

.. code:: bash

    <outputDir>
        │
        └──█ formatted.txt


Development
-----------

To perform in-line debugging of the container, do

.. code:: bash

    docker run --rm -it --userns=host  -u $(id -u):$(id -g)                                     \
        -v $PWD/unirep_analysis.py:/usr/local/lib/python3.5/dist-packages/unirep_analysis.py:ro \
        -v $PWD/src:/usr/local/lib/python3.5/dist-packages/src                                  \
           -v $PWD/in:/incoming:ro -v $PWD/out:/outgoing:rw -w /outgoing                        \
           local/pl-unirep_analysis2 unirep_analysis /incoming /outgoing

Note, if you want to use `pudb` for debugging, then omit the ``-u $(id -u):$(id -g)``:

.. code:: bash

    docker run --rm -it --userns=host                                                           \
        -v $PWD/unirep_analysis.py:/usr/local/lib/python3.5/dist-packages/unirep_analysis.py:ro \
        -v $PWD/src:/usr/local/lib/python3.5/dist-packages/src                                  \
           -v $PWD/in:/incoming:ro -v $PWD/out:/outgoing:rw -w /outgoing                        \
           local/pl-unirep_analysis2 unirep_analysis /incoming /outgoing

Of course, in both cases above, use approrpiate CLI args if required.

.. image:: https://raw.githubusercontent.com/FNNDSC/cookiecutter-chrisapp/master/doc/assets/badge/light.png
    :target: https://chrisstore.co

_-30-_