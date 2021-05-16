.. _reference:

Reference
=========

This is the class and function reference of cycompsense. Please refer to
the :ref:`tutorial <tutorial>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.

.. automodule:: compsense.operators

.. autosummary::
   :toctree: generated/

   compsense.operators.opBase
   compsense.operators.opMatrix
   compsense.operators.opBlur
   compsense.operators.opRandMask
   compsense.operators.opWavelet
   compsense.operators.opDirac
   compsense.operators.opFoG
   compsense.operators.opFFT2d
   compsense.operators.opDCT
   compsense.operators.op3DStack


.. automodule:: compsense.problems

.. autosummary::
   :toctree: generated/

   compsense.problems.problemBase
   compsense.problems.problemBase.reconstruct
   compsense.problems.probCustom
   compsense.problems.prob701
   compsense.problems.probMissingPixels


.. automodule:: compsense.algorithms

.. autosummary::
   :toctree: generated/

   compsense.algorithms.algorithmBase
   compsense.algorithms.algorithmBase.solve
   compsense.algorithms.TwIST
   compsense.algorithms.TwIST_raw
