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
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.HAN
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsContainer
import com.kotlinnlp.simplednn.utils.Serializer
import java.io.InputStream
import java.io.OutputStream

/**
 * The [HANClassifier] model.
 *
 * @param outputSize the size of the output
 * @param embeddingsSize the size of the embeddings (used also for the attention arrays, default = 50)
 * @param recurrentConnectionType the recurrent connection type of the recurrent neural networks
 */
class HANClassifierModel(
  outputSize: Int,
  embeddingsSize: Int = 50,
  recurrentConnectionType: LayerType.Connection = LayerType.Connection.GRU
) {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [HANClassifierModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [HANClassifierModel]
     *
     * @return the [HANClassifierModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): HANClassifierModel = Serializer.deserialize(inputStream)
  }

  /**
   * The embeddings associated to each token.
   */
  val embeddings = EmbeddingsContainer(count = 1e05.toInt(), size = embeddingsSize).initialize()

  /**
   * The [HAN] model of the encoder.
   */
  val han: HAN = HAN(
    hierarchySize = 2,
    inputSize = embeddingsSize,
    inputType = LayerType.Input.Dense,
    biRNNsActivation = Tanh(),
    biRNNsConnectionType = recurrentConnectionType,
    attentionSize = embeddingsSize,
    outputSize = outputSize,
    outputActivation = Softmax()).initialize()

  /**
   * Serialize this [HANClassifierModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [HANClassifierModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
