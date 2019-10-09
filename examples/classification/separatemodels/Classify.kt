/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package classification.separatemodels

import com.kotlinnlp.hanclassifier.HANClassifier
import com.kotlinnlp.hanclassifier.HANClassifierModel
import com.kotlinnlp.hanclassifier.LabelsConfig
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.neuraltokenizer.NeuralTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.kotlinnlp.tokensencoder.reduction.ReductionEncoderModel
import com.xenomachina.argparser.mainBody
import java.io.File
import java.io.FileInputStream

/**
 * Classify texts from standard input with a transient embeddings encoder, loading the embeddings separately.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val model: HANClassifierModel = parsedArgs.classifierModelPath.let {
    println("Loading HAN classifier model from '$it'...")
    HANClassifierModel.load(FileInputStream(File(it)))
  }

  val embeddingsMap: EmbeddingsMap<String> = parsedArgs.embeddingsPath.let {
    println("Loading pre-trained word embeddings from '$it'...")
    EmbeddingsMap.load(it)
  }

  val inputTokensEncoder: EmbeddingsEncoderModel.Transient<*, *> =
    (model.tokensEncoder as ReductionEncoderModel).inputEncoderModel as EmbeddingsEncoderModel.Transient

  inputTokensEncoder.setEmbeddingsMap(embeddingsMap)

  val classifier = HANClassifier(model)

  val tokenizer = NeuralTokenizer(model = parsedArgs.tokenizerModelPath.let {
    println("Loading tokenizer model from '$it'...")
    NeuralTokenizerModel.load(FileInputStream(File(it)))
  })

  val labelsConfig: LabelsConfig? = parsedArgs.labelsConfigPath?.let {
    println("Loading labels configuration from '$it'...")
    LabelsConfig.fromFile(it)
  }

  var inputText = readInput()

  while (inputText.isNotEmpty()) {

    @Suppress("UNCHECKED_CAST")
    val sentences: List<Sentence<FormToken>> = tokenizer.tokenize(inputText).map { it as Sentence<FormToken> }
    val sentencesToUse: List<Sentence<FormToken>> = if (parsedArgs.reduceSentences) reduce(sentences) else sentences
    val predictions: List<DenseNDArray> = classifier.classify(sentencesToUse)
    val confidencePerLevel: List<Double> = predictions.map { it.max() }
    val predictedClass: String =
      labelsConfig?.getLabel(predictions) ?: predictions.joinToString("-") { it.argMaxIndex().toString() }

    var accuracy = 1.0
    confidencePerLevel.forEach { accuracy *= it }

    println()
    println("Predicted class: '$predictedClass' (level: ${predictions.size})")
    println("Global confidence: %.1f%% (%s)"
      .format(100.0 * accuracy, confidencePerLevel.joinToString(" x ") { "%.1f%%".format(100.0 * it) }))

    inputText = readInput()
  }

  println("Done.")
}

/**
 * Read a text from the standard input.
 *
 * @return the string read
 */
private fun readInput(): String {

  print("\nClassify a text (empty to exit): ")

  return readLine()!!.trim()
}

/**
 * Reduce the sentences to the first two with at least 5 tokens.
 *
 * @param sentences a list of sentences
 *
 * @return a list of reduced sentences
 */
private fun reduce(sentences: List<Sentence<FormToken>>): List<Sentence<FormToken>> {

  var count = 0

  return sentences.takeWhile {
    if (it.tokens.size >= 5) count++
    count <= 2
  }
}
