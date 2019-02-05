/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The [HANClassifier] model.
 *
 * @property multiLevelHAN the model of a multi-level HAN structure
 * @property tokensEncoder the model of a tokens encoder to encode the input
 */
class HANClassifierModel(
  val multiLevelHAN: MultiLevelHANModel,
  val tokensEncoder: TokensEncoderModel<FormToken, Sentence<FormToken>>
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Build a [HANClassifierModel] without an explicit declaration of the [MultiLevelHANModel].
     *
     * @param name the name of the model
     * @param classesConfig the configurations of the hierarchy of classes that can be predicted
     * @param tokensEncoder the model of a tokens encoder to encode the input
     * @param attentionSize the size of the attention arrays (default = 20)
     * @param recurrentConnectionType the recurrent connection type of the recurrent neural networks
     */
    operator fun invoke(name: String,
                        classesConfig: ClassesConfig,
                        tokensEncoder: TokensEncoderModel<FormToken, Sentence<FormToken>>,
                        attentionSize: Int = 20,
                        recurrentConnectionType: LayerType.Connection = LayerType.Connection.GRU) = HANClassifierModel(
      multiLevelHAN = MultiLevelHANModel(
        name = name,
        classesConfig = classesConfig,
        tokenEncodingSize = tokensEncoder.tokenEncodingSize,
        attentionSize = attentionSize,
        recurrentConnectionType = recurrentConnectionType
      ),
      tokensEncoder = tokensEncoder
    )

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
   * The name of the model.
   */
  val name: String = this.multiLevelHAN.name

  /**
   * The configurations of the hierarchy of classes that can be predicted.
   */
  val classesConfig: ClassesConfig = this.multiLevelHAN.classesConfig

  /**
   * The classifiers models by level.
   */
  internal val topLevelModel: MultiLevelHANModel.LevelModel = this.multiLevelHAN.topLevelModel

  /**
   * Check requirements.
   */
  init {
    require(this.multiLevelHAN.tokenEncodingSize == this.tokensEncoder.tokenEncodingSize) {
      "The tokens encoding size of the TokensEncoder must be compatible with the MultiLevelHANModel."
    }
  }

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

    var classConfig: ClassesConfig? = this.classesConfig

    classHierarchy.forEach { classConfig = classConfig!!.classes[it] }

    return classConfig != null
  }

  /**
   * @param classHierarchy a list of classes following their hierarchical order
   *
   * @return the index of the 'no-class' of the hierarchical level defined by the given class hierarchy
   */
  internal fun getNoClassIndex(classHierarchy: List<Int>): Int {

    var classConfig: ClassesConfig = this.classesConfig

    classHierarchy.forEach { classConfig = classConfig.classes[it]!! }

    return classConfig.classes.size
  }
}
