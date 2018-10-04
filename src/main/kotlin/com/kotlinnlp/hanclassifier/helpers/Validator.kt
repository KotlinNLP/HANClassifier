/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier.helpers

import com.kotlinnlp.hanclassifier.EncodedSentence
import com.kotlinnlp.hanclassifier.HANClassifier
import com.kotlinnlp.hanclassifier.HANClassifierModel
import com.kotlinnlp.hanclassifier.dataset.Example
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar

/**
 * A helper for the validation of a [HANClassifier].
 *
 * @param model the model
 * @param tokensEncoder the tokens encoder used to encode the input
 */
class Validator(
  model: HANClassifierModel,
  private val tokensEncoder: TokensEncoder<FormToken, Sentence<FormToken>>
) {

  /**
   * The classifier initialized with the model.
   */
  val classifier = HANClassifier(
    model = model,
    useDropout = true,
    propagateToInput = true)

  /**
   * When timing started.
   */
  private var startTime: Long = 0

  /**
   * Validate the [classifier] using the given test dataset.
   *
   * @param testSet the test dataset to validate the [classifier]
   *
   * @return the accuracy of the [classifier]
   */
  fun validate(testSet: List<Example>): Double {

    var correctPredictions = 0
    val progress = ProgressIndicatorBar(testSet.size)

    this.startTiming()

    testSet.forEach { example ->

      progress.tick()

      correctPredictions += this.validateExample(example)
    }

    println("Elapsed time: %s".format(this.formatElapsedTime()))

    return correctPredictions.toDouble() / testSet.size
  }

  /**
   * Validate the HAN classifier with the given [example].
   *
   * @param example an example of the validation dataset
   *
   * @return 1 if the prediction is correct, 0 otherwise
   */
  private fun validateExample(example: Example): Int {

    val output: DenseNDArray = this.classifier.forward(
      input = example.sentences.map { EncodedSentence(this.tokensEncoder.forward(it)) })

    return if (this.predictionIsCorrect(output, example.outputGold)) 1 else 0
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
