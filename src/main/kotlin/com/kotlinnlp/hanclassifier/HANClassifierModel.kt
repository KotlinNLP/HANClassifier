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
import com.kotlinnlp.simplednn.deeplearning.attention.han.HAN
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The [HANClassifier] model.
 *
 * @param tokensEncodingsSize the size of the tokens encodings
 * @param outputSize the size of the output
 * @param attentionSize the size of the attention arrays (default = 20)
 * @param recurrentConnectionType the recurrent connection type of the recurrent neural networks
 */
class HANClassifierModel(
  tokensEncodingsSize: Int,
  outputSize: Int,
  attentionSize: Int = 20,
  recurrentConnectionType: LayerType.Connection = LayerType.Connection.GRU
) : Serializable {

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
   * The [HAN] model of the encoder.
   */
  val han = HAN(
    hierarchySize = 2,
    inputSize = tokensEncodingsSize,
    inputType = LayerType.Input.Dense,
    biRNNsActivation = Tanh(),
    biRNNsConnectionType = recurrentConnectionType,
    attentionSize = attentionSize,
    outputSize = outputSize,
    outputActivation = Softmax())

  /**
   * Serialize this [HANClassifierModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [HANClassifierModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
