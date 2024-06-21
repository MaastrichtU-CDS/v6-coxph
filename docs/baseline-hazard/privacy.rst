Privacy
=======

Guards
------

.. What have you done to protect your users' privacy? E.g. threshold on low counts,
.. noise addition, etc.

Data sharing
------------

.. which data is shared between the parties? E.g. for an average, sum and total count
.. are shared.

Vulnerabilities to known attacks
--------------------------------

.. Table below lists some well-known attacks. You could fill in this table to show
.. which attacks would be possible in your system.

.. list-table::
    :widths: 25 10 65
    :header-rows: 1

    * - Attack
      - Risk eliminated?
      - Risk analysis
    * - Reconstruction
      - ⚠
      - May happen if ...
    * - Differencing
      - ❌
      - Possible by doing A then B...
    * - Deep Leakage from Gradients (DLG)
      - ✔
      -
    * - Generative Adversarial Networks (GAN)
      - ✔
      -
    * - Model Inversion
      - ✔
      -
    * - Watermark Attack
      - ✔
      -