/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A classifier based on Hierarchic Attention Networks, that works on a hierarchical structure of classes.
 *
 * @param model the model of this [HANClassifier]
 * @param useDropout whether to apply the dropout during the forward (default = false)
 * @param propagateToInput whether to propagate the errors to the input during the backward (default = false)
 */
class HANClassifier(
  val model: HANClassifierModel,
  val useDropout: Boolean = false,
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
   * Classify the given [input].
   *
   * @param input a list of encoded sentences
   *
   * @return the list with the probability distribution of the classification, one per hierarchical level
   */
  fun classify(input: List<EncodedSentence>): List<DenseNDArray> =
    this.forwardLevel(input = input, levelClassifier = this.topLevelClassifier)

  /**
   * Classify the given [input].
   *
   * @param input a list of encoded sentences
   * @param levelClassifier the classifier for a given hierarchical level
   * @param levelIndex the index of the level
   *
   * @return the list with the probability distribution of the classification of the given level and its sub-levels
   */
  private fun forwardLevel(input: List<EncodedSentence>,
                           levelClassifier: LevelClassifier,
                           levelIndex: Int = 0): List<DenseNDArray> {

    val output: List<DenseNDArray> = listOf(levelClassifier.classifier.forward(input))
    val predictedClass: Int = output.single().argMaxIndex()
    val subLevelClassifier: HANClassifier.LevelClassifier? = levelClassifier.subLevels[predictedClass]

    return if (subLevelClassifier != null)
      output + this.forwardLevel(input = input, levelClassifier = subLevelClassifier, levelIndex = levelIndex + 1)
    else
      output
  }

  /**
   * @param levelModel the model of a level classifier
   *
   * @return a level classifier based on the given model
   */
  private fun buildLevelClassifier(levelModel: HANClassifierModel.LevelModel): LevelClassifier =
    LevelClassifier(
      classifier = HANClassifierSingle(
        han = levelModel.han,
        useDropout = this.useDropout,
        propagateToInput = this.propagateToInput),
      subLevels = levelModel.subLevels.mapValues { it.value?.let { model -> this.buildLevelClassifier(model) } })
}
