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
 * @param name the name of the model
 * @param classesConfig the configurations of the hierarchy of classes that can be predicted
 * @param tokensEncodingsSize the size of the tokens encodings
 * @param attentionSize the size of the attention arrays (default = 20)
 * @param recurrentConnectionType the recurrent connection type of the recurrent neural networks
 */
class HANClassifierModel(
  val name: String,
  classesConfig: ClassesConfig,
  private val tokensEncodingsSize: Int,
  private val attentionSize: Int = 20,
  private val recurrentConnectionType: LayerType.Connection = LayerType.Connection.GRU
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
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
   * The model of a hierarchical level.
   *
   * @property han a hierarchical attention network to classify a level
   * @property subLevels the map of hierarchical sub-levels models associated by class index (null if there is none)
   */
  internal data class LevelModel(val han: HAN, val subLevels: Map<Int, LevelModel?>) : Serializable {

    companion object {

      /**
       * Private val used to serialize the class (needed by Serializable).
       */
      @Suppress("unused")
      private const val serialVersionUID: Long = 1L
    }
  }

  /**
   * The classifiers models by level.
   */
  internal val topLevelModel: LevelModel = this.buildLevelModel(classesConfig)

  /**
   * Serialize this [HANClassifierModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [HANClassifierModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)

  /**
   * @param classHierarchy a list of classes following their hierarchical order
   *
   * @return true if the last level of the given hierarchy has sub-level itself, otherwise false
   */
  internal fun hasSubLevels(classHierarchy: List<Int>): Boolean {

    var levelModel: LevelModel? = this.topLevelModel

    classHierarchy.forEach { levelModel = levelModel!!.subLevels[it] }

    return levelModel != null
  }

  /**
   * @param config the configuration of classes of a given hierarchical level
   * @param level the hierarchical level (0 means the top)
   *
   * @return a hierarchical level model
   */
  private fun buildLevelModel(config: ClassesConfig, level: Int = 0): LevelModel = LevelModel(
    han = HAN(
      hierarchySize = 2,
      inputSize = this.tokensEncodingsSize,
      inputType = LayerType.Input.Dense,
      biRNNsActivation = Tanh(),
      biRNNsConnectionType = this.recurrentConnectionType,
      attentionSize = this.attentionSize,
      outputSize = config.classes.size + (if (level > 0) 1 else 0), // include the 'no-class' for lower levels
      outputActivation = Softmax()),
    subLevels = config.classes.entries
      .associate { it.key to it.value?.let { config -> this.buildLevelModel(config = config, level = level + 1) } }
  )
}
