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
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.deeplearning.attention.han.HANParameters
import com.kotlinnlp.tokensencoder.TokensEncoder
import com.kotlinnlp.tokensencoder.TokensEncoderOptimizer
import com.kotlinnlp.utils.ExamplesIndices
import com.kotlinnlp.utils.Shuffler
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import java.io.File
import java.io.FileOutputStream

/**
 * A helper for the training of a [HANClassifier].
 *
 * @property classifier the [HANClassifier] to train
 * @param tokensEncoder the tokens encoder to encode the input
 * @param tokensEncoderOptimizer the optimizer of the tokens encoder (null if the tokens encoder must not be trained)
 * @param updateMethod the update method for the parameters of the [classifier]
 * @param onSaveModel a callback called when saving a new best model
 */
class Trainer(
  private val classifier: HANClassifier,
  tokensEncoder: TokensEncoder<FormToken, Sentence<FormToken>>,
  private val tokensEncoderOptimizer: TokensEncoderOptimizer? = null,
  private val updateMethod: UpdateMethod<*>,
  private val onSaveModel: () -> Unit = {}
) {

  /**
   * The optimizer of a level classifier of the hierarchy.
   */
  private data class LevelOptimizer(
    val optimizer: ParamsOptimizer<HANParameters>,
    val subLevels: Map<Int, LevelOptimizer?>
  )

  /**
   * When timing started.
   */
  private var startTime: Long = 0

  /**
   * The best accuracy reached during the training.
   */
  private var bestAccuracy: Double = 0.0

  /**
   * The helper for the validation of the [classifier].
   */
  private val validationHelper = Validator(model = this.classifier.model, tokensEncoderModel = tokensEncoder.model)

  /**
   * A pool of tokens encoders to encode the input.
   */
  private val tokensEncodersPool = TokensEncodersPool(
    model = tokensEncoder.model,
    useDropout = tokensEncoder.useDropout)

  /**
   * The list of all the HAN optimizers of all the levels.
   * It is filled calling the [buildLevelOptimizer] method.
   */
  private val classifierOptimizers: MutableList<ParamsOptimizer<HANParameters>> = mutableListOf()

  /**
   * The optimizer of the parameters of the top level encoder of the [classifier].
   */
  private val topLevelOptimizer: LevelOptimizer = this.buildLevelOptimizer(this.classifier.model.topLevelModel)

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
   * Build the optimizer of the HAN of a given hierarchical level.
   *
   * @param levelModel the model of a hierarchical level
   *
   * @return a level optimizer
   */
  private fun buildLevelOptimizer(levelModel: HANClassifierModel.LevelModel): LevelOptimizer {

    val levelOptimizer = LevelOptimizer(
      optimizer = ParamsOptimizer(params = levelModel.han.params, updateMethod = this.updateMethod),
      subLevels = levelModel.subLevels.mapValues { it.value?.let { model -> this.buildLevelOptimizer(model) } })

    this.classifierOptimizers.add(levelOptimizer.optimizer)

    return levelOptimizer
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

    val encoders: List<TokensEncoder<FormToken, Sentence<FormToken>>> = this.getTokensEncoders(example.sentences.size)

    val encodedSentences: List<EncodedSentence> =
      example.sentences.zip(encoders) { sentence, encoder -> EncodedSentence(encoder.forward(sentence)) }

    val sentencesErrors: List<EncodedSentence> =
      encodedSentences.map { s -> s.copy(tokens = s.tokens.map { it.zerosLike() }) }

    this.trainLevelClassifier(
      levelClassifier = this.classifier.topLevelClassifier,
      levelOptimizer = this.topLevelOptimizer,
      encodedSentences = encodedSentences,
      sentencesErrors = sentencesErrors,
      expectedClasses = if (this.classifier.model.hasSubLevels(example.goldClasses))
        example.goldClasses + this.classifier.model.getNoClassIndex(example.goldClasses)
      else
        example.goldClasses
    )

    this.tokensEncoderOptimizer?.let { optimizer ->
      encoders.zip(sentencesErrors).forEach { (encoder, sentenceErrors) ->
        encoder.backward(sentenceErrors.tokens)
        optimizer.accumulate(encoder.getParamsErrors())
      }
    }
  }

  /**
   * @param size the number of tokens encoder to return
   *
   * @return a list of tokens encoders got from the pool
   */
  private fun getTokensEncoders(size: Int): List<TokensEncoder<FormToken, Sentence<FormToken>>> {

    this.tokensEncodersPool.releaseAll()

    return List(size = size, init = { this.tokensEncodersPool.getItem() })
  }

  /**
   * Train the classifier of a specific level of the classes hierarchy.
   *
   * @param levelClassifier the level classifier
   * @param levelOptimizer the level optimizer
   * @param encodedSentences the input encoded sentences
   * @param sentencesErrors the structures in which to accumulate the errors of the sentences
   * @param expectedClasses the list of expected classes, following the hierarchical order from the top level
   * @param levelIndex the index of the current level, as depth in the hierarchy
   */
  private fun trainLevelClassifier(levelClassifier: HANClassifier.LevelClassifier,
                                   levelOptimizer: LevelOptimizer,
                                   encodedSentences: List<EncodedSentence>,
                                   sentencesErrors: List<EncodedSentence>,
                                   expectedClasses: List<Int>,
                                   levelIndex: Int = 0) {

    val expectedClass: Int = expectedClasses[levelIndex]
    val distribution: DenseNDArray = levelClassifier.classifier.forward(encodedSentences)

    val errors: DenseNDArray = distribution.copy()
    errors[expectedClass] = errors[expectedClass] - 1

    levelClassifier.classifier.backward(errors)
    levelOptimizer.optimizer.accumulate(levelClassifier.classifier.getParamsErrors(copy = false))

    if (this.tokensEncoderOptimizer != null) {
      levelClassifier.classifier.getInputErrors(copy = false).zip(sentencesErrors) { inputErrors, sentenceErrors ->
        sentenceErrors.tokens.zip(inputErrors.tokens).forEach { it.first.assignSum(it.second) }
      }
    }

    if (levelIndex < expectedClasses.lastIndex)
      this.trainLevelClassifier(
        levelClassifier = levelClassifier.subLevels.getValue(expectedClass)!!,
        levelOptimizer = levelOptimizer.subLevels.getValue(expectedClass)!!,
        encodedSentences = encodedSentences,
        sentencesErrors = sentencesErrors,
        expectedClasses = expectedClasses,
        levelIndex = levelIndex + 1)
  }

  /**
   * Method to call every new epoch.
   */
  private fun newEpoch() {
    this.classifierOptimizers.forEach { it.newEpoch() }
    this.tokensEncoderOptimizer?.newEpoch()
  }

  /**
   * Method to call every new batch.
   */
  private fun newBatch() {
    this.classifierOptimizers.forEach { it.newBatch() }
    this.tokensEncoderOptimizer?.newBatch()
  }

  /**
   * Method to call every new example.
   */
  private fun newExample() {
    this.classifierOptimizers.forEach { it.newExample() }
    this.tokensEncoderOptimizer?.newExample()
  }

  /**
   * Optimizers update.
   */
  private fun update() {
    this.classifierOptimizers.forEach { it.update() }
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

      this.onSaveModel()
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
