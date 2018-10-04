/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier.helpers

import com.kotlinnlp.hanclassifier.EncodedSentence
import com.kotlinnlp.hanclassifier.HANClassifier
import com.kotlinnlp.hanclassifier.dataset.Example
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.dataset.Shuffler
import com.kotlinnlp.simplednn.helpers.training.utils.ExamplesIndices
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.deeplearning.attention.han.HANParameters
import com.kotlinnlp.tokensencoder.TokensEncoder
import com.kotlinnlp.tokensencoder.TokensEncoderOptimizer
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import java.io.File
import java.io.FileOutputStream

/**
 * A helper for the training of a [HANClassifier].
 *
 * @property classifier the [HANClassifier] to train
 * @param updateMethod the [UpdateMethod] for the parameters of the [classifier]
 * @param tokensEncoder the tokens encoder used to encode the input
 * @param tokensEncoderOptimizer the optimizer of the tokens encoder (null if the tokens encoder must not be trained)
 */
class Trainer(
  private val classifier: HANClassifier,
  updateMethod: UpdateMethod<*>,
  private val tokensEncoder: TokensEncoder<FormToken, Sentence<FormToken>>,
  private val tokensEncoderOptimizer: TokensEncoderOptimizer? = null
) {

  /**
   * When timing started.
   */
  private var startTime: Long = 0

  /**
   * The best accuracy reached during the training.
   */
  private var bestAccuracy: Double = 0.0

  /**
   * The helper for the valdiation of the [classifier].
   */
  private val validationHelper = Validator(model = this.classifier.model, tokensEncoder = this.tokensEncoder)

  /**
   * The optimizer of the parameters of the [classifier].
   */
  private val classifierOptimizer: ParamsOptimizer<HANParameters> = ParamsOptimizer(
    params = this.classifier.model.han.params,
    updateMethod = updateMethod)

  /**
   * Train the [classifier] using the given [trainingSet], validating each epoch if a [validationSet] is given.
   *
   * @param trainingSet the dataset to train the classifier
   * @param epochs the number of epochs for the training
   * @param batchSize the size of each batch of examples (default = 1)
   * @param shuffler the [Shuffler] to shuffle the training sentences before each epoch (can be null)
   * @param validationSet the dataset to validate the classifier after each epoch (default = null)
   * @param modelFilename the name of the file in which to save the best trained model (default = null)
   */
  fun train(trainingSet: List<Example>,
            epochs: Int,
            batchSize: Int = 1,
            shuffler: Shuffler? = Shuffler(enablePseudoRandom = true, seed = 743),
            validationSet: List<Example>? = null,
            modelFilename: String? = null) {

    (0 until epochs).forEach { i ->

      println("\nEpoch ${i + 1} of $epochs")

      this.startTiming()

      this.newEpoch()
      this.trainEpoch(trainingSet = trainingSet, batchSize = batchSize, shuffler = shuffler)

      println("Elapsed time: %s".format(this.formatElapsedTime()))

      if (validationSet != null) {
        this.validateAndSaveModel(validationSet = validationSet, modelFilename = modelFilename)
      }
    }
  }

  /**
   * Train the HAN classifier on one epoch.
   *
   * @param trainingSet the training set
   * @param batchSize the size of each batch of examples
   * @param shuffler the [Shuffler] to shuffle examples before training (can be null)
   */
  private fun trainEpoch(trainingSet: List<Example>,
                         batchSize: Int,
                         shuffler: Shuffler?) {

    val progress = ProgressIndicatorBar(trainingSet.size)
    var examplesCount = 0

    for (exampleIndex in ExamplesIndices(size = trainingSet.size, shuffler = shuffler)) {

      examplesCount++
      progress.tick()

      if ((examplesCount - 1) % batchSize == 0) {
        this.newBatch()
      }

      this.newExample()
      this.learnFromExample(example = trainingSet[exampleIndex])

      if (examplesCount % batchSize == 0 || examplesCount == trainingSet.size) {
        this.update()
      }
    }
  }

  /**
   * Learn from the given [example], comparing its gold output class with the one of the [classifier] and accumulate
   * the propagated errors.
   *
   * @param example the example from which to learn
   */
  private fun learnFromExample(example: Example) {

    val output: DenseNDArray = this.classifier.forward(
      input = example.sentences.map { EncodedSentence(this.tokensEncoder.forward(it)) })

    val errors: DenseNDArray = output.copy()
    errors[example.outputGold] = errors[example.outputGold] - 1

    this.classifier.backward(errors)
    this.classifierOptimizer.accumulate(this.classifier.getParamsErrors(copy = false))

    this.tokensEncoderOptimizer?.let { optimizer ->
      this.classifier.getInputErrors(copy = false).forEach {
        this.tokensEncoder.backward(it.tokens)
        optimizer.accumulate(this.tokensEncoder.getParamsErrors())
      }
    }
  }

  /**
   * Method to call every new epoch.
   */
  private fun newEpoch() {
    this.classifierOptimizer.newEpoch()
    this.tokensEncoderOptimizer?.newEpoch()
  }

  /**
   * Method to call every new batch.
   */
  private fun newBatch() {
    this.classifierOptimizer.newBatch()
    this.tokensEncoderOptimizer?.newBatch()
  }

  /**
   * Method to call every new example.
   */
  private fun newExample() {
    this.classifierOptimizer.newExample()
    this.tokensEncoderOptimizer?.newExample()
  }

  /**
   * Optimizers update.
   */
  private fun update() {
    this.classifierOptimizer.update()
    this.tokensEncoderOptimizer?.update()
  }

  /**
   * Validate the [classifier] on the [validationSet] and save its best model to [modelFilename].
   *
   * @param validationSet the validation dataset to validate the [classifier]
   * @param modelFilename the name of the file in which to save the best model of the [classifier] (default = null)
   */
  private fun validateAndSaveModel(validationSet: List<Example>, modelFilename: String?) {

    val accuracy = this.validateEpoch(validationSet)

    println("Accuracy: %.2f%%".format(100.0 * accuracy))

    if (modelFilename != null && accuracy > this.bestAccuracy) {

      this.bestAccuracy = accuracy

      this.classifier.model.dump(FileOutputStream(File(modelFilename)))

      println("NEW BEST ACCURACY! Model saved to \"$modelFilename\"")
    }
  }

  /**
   * Validate the [classifier] after trained it on an epoch.
   *
   * @param validationSet the validation dataset to validate the [classifier]
   *
   * @return the current accuracy of the [classifier]
   */
  private fun validateEpoch(validationSet: List<Example>): Double {

    println("Epoch validation on %d sentences".format(validationSet.size))

    return this.validationHelper.validate(validationSet)
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
