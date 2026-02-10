tutorial
========

This is a page that provides a tutorial on how to use this package. This is by far not complete and currently serves mainly as a placeholder.

First you have to import the package::

    import darclight as dl

After that a basic data reduction process can look like this::

    data = dl.DataCollection('testdata', 'testdata/reduced')
    reducer = dl.Reducer(data)

    reducer.create_master_bias()

The last function creates a master bias from all detected frames in the provided folder of the DataCollection.