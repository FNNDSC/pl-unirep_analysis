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

        python unirep_analysis.py                                           \
                                    [--dimension <modelDimension>]          \
                                    [--batch_size <batchSize>]              \
                                    [--inputFile <inputFileToProcess>]      \
                                    [--outputFile <resultOutputFile>]       \
                                    [--train_top_model]                     \
                                    [--train_multiple]                      \
                                    [--json]                                \
                                    <inputDir>
                                    <outputDir>

Description
-----------

``unirep_analysis.py`` is a ChRIS-based application that is capable of training, inferencing representations, generative modelling aka "babbling", and data management

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
        to 1900 (full) or 256 and the corresponding weights files will be downloaded 
        AWS for you.
        
        [--batch_size <batchSize>]
        This represents the batch size of the babbler. Default value is 12.

        [--inputFile <inputFileToProcess>]
        The name of the ``.txt`` file that contains your amino acid sequences.
        The default file name is ``sequence.txt``. The full path to the 
        <inputFileToProcess> is constructed by concatenating <inputDir>

                ``<inputDir>/<inputFileToProcess>``

        [--outputFile <resultOutputFile>]
        The name of the output or formatted ``txt`` file. Default name is

                            ``format.txt``

        [--train_top_model]
        If specified, train top model only.



        [--train_multiple]
        If specified, jointly train top model & mLSTM
        
        (Note that if using the 1900-unit (full) model, 
        you will need a GPU with at least 16GB RAM. 
        To see a demonstration of joint training with fewer
        computational resources, please run this plugin using the 64-unit model.

        [-h]
        Display inline help

        [--json]
        If specified, print a JSON representation of the app.

Run
----

The execute vector of this pluing is via ``docker``.

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
                




.. image:: https://raw.githubusercontent.com/FNNDSC/cookiecutter-chrisapp/master/doc/assets/badge/light.png
    :target: https://chrisstore.co

