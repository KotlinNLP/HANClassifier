/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.*
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsContainer
import java.io.File
import java.io.FileInputStream

/**
 * A classifier based on Hierarchic Attention Networks.
 *
 * @param outputSize the size of the output
 * @param embeddingsSize the size of the embeddings (used also for the attention arrays, default = 50)
 * @param recurrentConnectionType the recurrent connection type of the recurrent neural networks
 * @param modelFilename the file name from which to load the model of the [HAN] (if null a new one is created)
 */
class HANClassifier(
  outputSize: Int,
  embeddingsSize: Int = 50,
  recurrentConnectionType: LayerType.Connection = LayerType.Connection.GRU,
  modelFilename: String? = null
) {

  /**
   * The embeddings associated to each token.
   */
  val embeddings = EmbeddingsContainer(count = 1e05.toInt(), size = embeddingsSize).randomize()

  /**
   * The [HAN] model of the encoder.
   */
  val model: HAN = if (modelFilename != null) HAN.load(FileInputStream(File(modelFilename))) else HAN(
    hierarchySize = 2,
    inputSize = embeddingsSize,
    biRNNsActivation = Tanh(),
    biRNNsConnectionType = recurrentConnectionType,
    attentionSize = embeddingsSize,
    outputSize = outputSize,
    outputActivation = Softmax()).initialize()

  /**
   * The [HANEncoder] used as classifier (Softmax output activation).
   */
  val encoder: HANEncoder = HANEncoder(model = this.model)
}
