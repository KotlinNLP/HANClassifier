/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package evaluation.separatemodels

import com.kotlinnlp.hanclassifier.HANClassifierModel
import com.kotlinnlp.hanclassifier.dataset.CorpusReader
import com.kotlinnlp.hanclassifier.dataset.Example
import com.kotlinnlp.hanclassifier.helpers.Validator
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.kotlinnlp.tokensencoder.reduction.ReductionEncoderModel
import com.xenomachina.argparser.mainBody
import java.io.File
import java.io.FileInputStream

/**
 * Evaluate a [HANClassifierModel] with a transient embeddings encoder, loading the embeddings separately.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val model: HANClassifierModel = parsedArgs.modelPath.let {
    println("Loading HAN classifier model from '$it'...")
    HANClassifierModel.load(FileInputStream(File(it)))
  }

  val embeddingsMap: EmbeddingsMap<String> = parsedArgs.embeddingsPath.let {
    println("Loading pre-trained word embeddings from '$it'...")
    EmbeddingsMap.load(it)
  }

  val validationSet: List<Example> = parsedArgs.validationSetPath.let {
    println("Loading validation dataset from '$it'...")
    CorpusReader().read(it)
  }

  val inputTokensEncoder: EmbeddingsEncoderModel.Transient<*, *> =
    (model.tokensEncoder as ReductionEncoderModel).inputEncoderModel as EmbeddingsEncoderModel.Transient

  inputTokensEncoder.setEmbeddingsMap(embeddingsMap)

  println("\n-- START VALIDATION ON %d SENTENCES".format(validationSet.size))

  val info: Validator.ValidationInfo = Validator(model).validate(testSet = validationSet)
  val accuracy: Double = info.metrics.map { it.f1Score }.average()

  println()
  println("Accuracy (f1 average): %5.2f %%".format(100 * accuracy))
  info.metrics.forEachIndexed { i, it -> println("- Level $i: $it") }

  println()
  println("Level 0 confusion:")
  println(info.confusionMatrix)
}
