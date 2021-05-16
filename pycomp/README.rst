======================
README for pycompsense
======================

`pycompsesne` is a toolbox for compressed sensing and sparse reconstruction algorithms.
It is based on `sparco <http://www.cs.ubc.ca/labs/scl/sparco/>`_.

`pycompsense` includes an implementation of `TwIST <http://www.lx.it.pt/~bioucas/TwIST/TwIST.htm>`_.


Installing
==========

Use ``setup.py``::

   python setup.py install


Reading the docs
================

After installing::

   cd doc
   make html

Then, direct your browser to ``build/html/index.html``.


Testing
=======

To run the tests with the interpreter available as ``python``, use::

   cd examples
   python demo_wave_DWT_deconv.py


Conditions for use
==================

pycompsense is open-source code released under the GNU Public License <http://www.gnu.org/copyleft/gpl.html>.


Contributing
============

For bug reports use the Bitbucket issue tracker.
You can also send wishes, comments, patches, etc. to amitibo@tx.technion.ac.il


Acknowledgement
===============

Thank-you to the people at <http://wingware.com/> for their policy of **free licenses for non-commercial open source developers**.

.. image:: http://wingware.com/images/wingware-logo-180x58.png
