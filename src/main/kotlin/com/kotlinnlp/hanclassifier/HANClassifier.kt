/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier

import com.kotlinnlp.hanclassifier.helpers.TokensEncodersPool
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A classifier based on Hierarchic Attention Networks, that works on a hierarchical structure of classes.
 *
 * @param model the model of this [HANClassifier]
 * @param biRNNDropout the probability of dropout for the BiRNNs (default 0.0)
 * @param attentionDropout the probability of attention dropout (default 0.0)
 * @param outputDropout the probability of output dropout (default 0.0)
 * @param propagateToInput whether to propagate the errors to the input during the backward (default = false)
 */
class HANClassifier(
  val model: HANClassifierModel,
  private val biRNNDropout: Double = 0.0,
  private val attentionDropout: Double = 0.0,
  private val outputDropout: Double = 0.0,
  val propagateToInput: Boolean = false
) {

  /**
   * The classifier of a single level of the hierarchy.
   *
   * @property classifier a single HAN classifier
   * @property subLevels the classifiers of the sub-levels of this one, associated by class index
   */
  internal data class LevelClassifier(val classifier: HANClassifierSingle, val subLevels: Map<Int, LevelClassifier?>)

  /**
   * The classifier used to classify the top level.
   */
  internal val topLevelClassifier: LevelClassifier = this.buildLevelClassifier(this.model.topLevelModel)

  /**
   * A pool of tokens encoders to encode the input.
   */
  private val tokensEncodersPool = TokensEncodersPool(model = this.model.tokensEncoder)

  /**
   * Classify the given [sentences].
   *
   * @param sentences a list of encoded sentences
   *
   * @return the list with the probability distribution of the classification, one per hierarchical level
   */
  fun classify(sentences: List<Sentence<FormToken>>): List<DenseNDArray> {

    this.tokensEncodersPool.releaseAll()

    return this.forwardLevel(
      input = sentences.map { EncodedSentence(this.tokensEncodersPool.getItem().forward(it)) },
      levelClassifier = this.topLevelClassifier)
  }

  /**
   * Classify the given [input].
   *
   * @param input a list of sentences
   * @param levelClassifier the classifier for a given hierarchical level
   * @param levelIndex the index of the level
   *
   * @return the list with the probability distribution of the classification of the given level and its sub-levels
   */
  private fun forwardLevel(input: List<EncodedSentence>,
                           levelClassifier: LevelClassifier,
                           levelIndex: Int = 0): List<DenseNDArray> {

    val prediction: DenseNDArray = levelClassifier.classifier.forward(input)
    val output: List<DenseNDArray> = listOf(prediction)

    val predictedClass: Int = prediction.argMaxIndex()
    val subLevelClassifier: LevelClassifier? = levelClassifier.subLevels[predictedClass]
    val goToSubLevel: Boolean = levelIndex == 0 || predictedClass < prediction.lastIndex // lastIndex == 'stop-level'

    return if (subLevelClassifier != null && goToSubLevel)
      output + this.forwardLevel(input = input, levelClassifier = subLevelClassifier, levelIndex = levelIndex + 1)
    else
      output
  }

  /**
   * @param levelModel the model of a level classifier
   *
   * @return a level classifier based on the given model
   */
  private fun buildLevelClassifier(levelModel: MultiLevelHANModel.LevelModel): LevelClassifier =
    LevelClassifier(
      classifier = HANClassifierSingle(
        han = levelModel.han,
        biRNNDropout = this.biRNNDropout,
        attentionDropout = this.attentionDropout,
        outputDropout = this.outputDropout,
        propagateToInput = this.propagateToInput),
      subLevels = levelModel.subLevels.mapValues { it.value?.let { model -> this.buildLevelClassifier(model) } })
}
