/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier.helpers

import com.kotlinnlp.hanclassifier.*
import com.kotlinnlp.hanclassifier.dataset.Example
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import com.kotlinnlp.utils.stats.MetricCounter

/**
 * A helper for the validation of a [HANClassifier].
 *
 * @param model the [HANClassifier] model to validate
 * @param tokensEncoderModel the model of a tokens encoder to encode the input
 */
class Validator(
  model: HANClassifierModel,
  tokensEncoderModel: TokensEncoderModel<FormToken, Sentence<FormToken>>
) {

  /**
   * A pool of tokens encoders to encode the input.
   */
  private val tokensEncodersPool = TokensEncodersPool(model = tokensEncoderModel, useDropout = false)

  /**
   * The classifier initialized with the model.
   */
  private val classifier = HANClassifier(
    model = model,
    useDropout = false,
    propagateToInput = false)

  /**
   * The metric counters per hierarchical level.
   */
  private val metricsPerLevel: List<MetricCounter> = List(size = model.classesConfig.depth, init = { MetricCounter() })

  /**
   * When timing started.
   */
  private var startTime: Long = 0

  /**
   * Validate the [classifier] using the given test dataset.
   *
   * @param testSet the test dataset to validate the [classifier]
   *
   * @return a list of metric counters, one for each hierarchical level
   */
  fun validate(testSet: List<Example>): List<MetricCounter> {

    val progress = ProgressIndicatorBar(testSet.size)

    this.startTiming()

    testSet.forEach { example ->

      progress.tick()

      this.validateExample(example)
    }

    println("Elapsed time: %s".format(this.formatElapsedTime()))

    return this.metricsPerLevel
  }

  /**
   * Validate the HAN classifier with the given [example].
   *
   * @param example an example of the validation dataset
   */
  private fun validateExample(example: Example) {

    val encoders: List<TokensEncoder<FormToken, Sentence<FormToken>>> =
      example.sentences.map { this.tokensEncodersPool.getItem() }

    val predictions: List<DenseNDArray> = this.classifier.classify(
      input = example.sentences.zip(encoders) { sentence, encoder -> EncodedSentence(encoder.forward(sentence)) })

    example.goldClasses.forEachIndexed { levelIndex, goldClass ->

      val metric: MetricCounter = this.metricsPerLevel[levelIndex]

      when {
        levelIndex < predictions.size -> metric.falseNeg++
        this.predictionIsCorrect(predictions[levelIndex], goldClass) -> metric.truePos++
        else -> metric.falsePos++
      }
    }
  }

  /**
   * @param output an output prediction of the HAN classifier
   * @param goldOutput the expected gold output class
   *
   * @return a Boolean indicating if the [output] matches the [goldOutput]
   */
  private fun predictionIsCorrect(output: DenseNDArray, goldOutput: Int): Boolean {
    return output.argMaxIndex() == goldOutput
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
