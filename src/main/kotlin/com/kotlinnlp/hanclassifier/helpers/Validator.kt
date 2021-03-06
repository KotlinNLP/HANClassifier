/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier.helpers

import com.kotlinnlp.hanclassifier.*
import com.kotlinnlp.hanclassifier.dataset.Example
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.ConfusionMatrix
import com.kotlinnlp.utils.progressindicator.ProgressIndicator
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import com.kotlinnlp.utils.stats.MetricCounter

/**
 * A helper for the validation of a [HANClassifier].
 *
 * @param model the [HANClassifier] model to validate
 * @param verbose whether to print progress information (default = true)
 */
class Validator(private val model: HANClassifierModel, private val verbose: Boolean = true) {

  /**
   * Validation info.
   *
   * @property metrics the metric counters per hierarchical level
   * @property confusionMatrix the confusion matrix of the first hierarchical level
   */
  inner class ValidationInfo(
    val metrics: List<MetricCounter> = List(size = model.classesConfig.depth, init = { MetricCounter() }),
    val confusionMatrix: ConfusionMatrix = ConfusionMatrix(
      labels = List(size = this.model.classesConfig.classes.size, init = { i -> i.toString() }))
  )

  /**
   * The classifier initialized with the model.
   */
  private val classifier = HANClassifier(model = this.model, propagateToInput = false)

  /**
   * The info filled during the current validation.
   */
  private lateinit var validationInfo: ValidationInfo

  /**
   * When timing started.
   */
  private var startTime: Long = 0

  /**
   * Validate the [classifier] using the given test dataset.
   *
   * @param testSet the test dataset to validate the [classifier]
   *
   * @return the validation info
   */
  fun validate(testSet: List<Example>): ValidationInfo {

    val progress: ProgressIndicator? = if (this.verbose) ProgressIndicatorBar(testSet.size) else null

    this.startTiming()
    this.validationInfo = ValidationInfo()

    testSet.forEach { example ->

      progress?.tick()

      this.validateExample(example)
    }

    if (this.verbose) println("Elapsed time: %s".format(this.formatElapsedTime()))

    return this.validationInfo
  }

  /**
   * Validate the HAN classifier with the given [example].
   *
   * @param example an example of the validation dataset
   */
  private fun validateExample(example: Example) {

    val predictions: List<DenseNDArray> = this.classifier.classify(example.sentences)

    val expectedClasses: List<Int> = if (this.classifier.model.hasSubLevels(example.goldClasses))
      example.goldClasses + this.classifier.model.getNoClassIndex(example.goldClasses)
    else
      example.goldClasses

    expectedClasses.forEachIndexed { levelIndex, goldClass ->

      val metric: MetricCounter = this.validationInfo.metrics[levelIndex]
      val isNoClass: Boolean =
        levelIndex in 1..predictions.lastIndex && goldClass == (predictions[levelIndex].length - 1)

      if (levelIndex == 0)
        this.validationInfo.confusionMatrix.increment(expected = goldClass, found = predictions[0].argMaxIndex())

      when {
        levelIndex > predictions.lastIndex -> metric.falseNeg++
        this.predictionIsCorrect(predictions[levelIndex], goldClass) -> if (!isNoClass) metric.truePos++
        isNoClass -> metric.falseNeg++
        else -> metric.falsePos++
      }
    }
  }

  /**
   * @param prediction a prediction of the classifier
   * @param goldClass the expected class
   *
   * @return true if the [prediction] indicates the [goldClass], otherwise false
   */
  private fun predictionIsCorrect(prediction: DenseNDArray, goldClass: Int): Boolean {
    return prediction.argMaxIndex() == goldClass
  }

  /**
   * Start registering time.
   */
  private fun startTiming() {
    this.startTime = System.currentTimeMillis()
  }

  /**
   * @return the formatted string with elapsed time in seconds and minutes.
   */
  private fun formatElapsedTime(): String {

    val elapsedTime = System.currentTimeMillis() - this.startTime
    val elapsedSecs = elapsedTime / 1000.0

    return "%.3f s (%.1f min)".format(elapsedSecs, elapsedSecs / 60.0)
  }
}
