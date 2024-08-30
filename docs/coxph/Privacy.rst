Privacy
=======

Guards
------

Binning
~~~~~~~~
The algorithm uses binning to protect the privacy of the unique event times shared by data stations.
The optimal number of bins is based on Sturges' rule and can binning can then be used in two ways:
With fixed binning events are divided into equal-sized;
With quantile binning events are categorized based on an optimal number of quantiles.


Differential privacy
~~~~~~~~~~~~~~~~~~~~
The algorithm has differential privacy built in and it can separately be enabled if the user wishes to do so.
It can be enabled for aggregates or predictors, which has different implications.
The differential privacy samples noise from a Laplacian distribution for the aggregates and continuous predictors, whilst the randomized response technique is used for the categorical predictors.
The user can specify the degree of differential privacy by setting the epsilon and sensitivity value.

Sample size threshold
~~~~~~~~~~~~~~~~~~~~~
The algorithm has a minimal threshold for the number of rows in the selected database. This threshold is set to 10 rows.
If the number of rows in a given data station is below this threshold,
the data station will not be included in the federated learning process and will be marked in the result.
This is determined in the first partial task.
This measure is identifiable as the 'N-threshold' in the central and partial functions.


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
      - This is not possible when using the incorporated differential privacy on the aggregates.
    * - Generative Adversarial Networks (GAN)
      - ✔
      - Synthetic can indeed be used to (statistically) reproduce the data that underlies the produced model, but without knowing the sensitive information the adversary will not be able to assess its authenticity. Using binning and/or differential privacy will further reduce this risk.
    * - Model Inversion
      - ✔
      - The model prediction can indeed be used to infer the outcome of an actual individual, but without knowing the sensitive information the adversary will not be able to assess its authenticity. Using binning and/or differential privacy will further reduce this risk.
    * - Watermark Attack
      - ⚠
      - To be determined

.. TODO verify whether these definitions are correct.
For reference:

- Reconstruction: This attack involves an adversary trying to reconstruct the original dataset from the shared model parameters. This is a risk if the model reveals too much information about the data it was trained on.
- Differencing: This attack involves an adversary trying to infer information about a specific data point by comparing the outputs of a model trained with and without that data point.
- Deep Leakage from Gradients (DLG): In this attack, an adversary tries to infer the training data from the shared gradient updates during the training process. This is a risk in federated learning where model updates are shared between participants.
- Generative Adversarial Networks (GAN): This is not an attack per se, but GANs can be used by an adversary to generate synthetic data that is statistically similar to the original data, potentially revealing sensitive information.
- Model Inversion: This attack involves an adversary trying to infer the input data given the output of a model. In a federated learning context, this could be used to infer sensitive information from the model's predictions.
- Watermark Attack: This attack involves an adversary embedding a "watermark" in the model during training, which can later be used to identify the model or the data it was trained on. This is a risk in federated learning where multiple parties contribute to the training of a model.
