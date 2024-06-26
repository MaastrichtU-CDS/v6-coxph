Privacy
=======

Guards
------
.. TODO check whether n is >= 11

.. What have you done to protect your users' privacy? E.g. threshold on low counts,
.. noise addition, etc.

Data sharing
------------

The data stations share the unique event times with the central aggregator, but, event times are not shared between data stations themselves.
The aggregated model parameters (coefficients) are shared with data stations by the central aggregator during the iteration process.

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
      - ✔
      - The amount of information shared was considered insufficient to allow reconstruction of the data underlying the model.
    * - Differencing
      - ⚠
      - This is indeed possible in case a data station manager were to change the dataset after performing a task, but data station managers should not be allowed to run tasks to prevent this.
    * - Deep Leakage from Gradients (DLG)
      - ✔
      - This is possible in the central aggregator, but this should be a trusted party and the shared information was considered insufficient to allow for DLG.
    * - Generative Adversarial Networks (GAN)
      - ✔
      - Synthetic can indeed be used to (statistically) reproduce the data that underlies the produced model, but without knowing the sensitive information the adversary will not be able to assess its authenticity.
    * - Model Inversion
      - ✔
      - The model prediction can indeed be used to infer the outcome of an actual individual, but without knowing the sensitive information the adversary will not be able to assess its authenticity.
    * - Watermark Attack
      - ✔
      - To be determined

.. TODO verify whether these definitions are correct.
For reference:

- Reconstruction: This attack involves an adversary trying to reconstruct the original dataset from the shared model parameters. This is a risk if the model reveals too much information about the data it was trained on.
- Differencing: This attack involves an adversary trying to infer information about a specific data point by comparing the outputs of a model trained with and without that data point.
- Deep Leakage from Gradients (DLG): In this attack, an adversary tries to infer the training data from the shared gradient updates during the training process. This is a risk in federated learning where model updates are shared between participants.
- Generative Adversarial Networks (GAN): This is not an attack per se, but GANs can be used by an adversary to generate synthetic data that is statistically similar to the original data, potentially revealing sensitive information.
- Model Inversion: This attack involves an adversary trying to infer the input data given the output of a model. In a federated learning context, this could be used to infer sensitive information from the model's predictions.
- Watermark Attack: This attack involves an adversary embedding a "watermark" in the model during training, which can later be used to identify the model or the data it was trained on. This is a risk in federated learning where multiple parties contribute to the training of a model.
