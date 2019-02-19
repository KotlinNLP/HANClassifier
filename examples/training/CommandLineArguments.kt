/* Copyright 2016-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package training

import com.xenomachina.argparser.ArgParser

/**
 * The interpreter of command line arguments.
 *
 * @param args the array of command line arguments
 */
internal class CommandLineArguments(args: Array<String>) {

  /**
   * The parser of the string arguments.
   */
  private val parser = ArgParser(args)

  /**
   * The name of the model.
   */
  val modelName: String by parser.storing(
    "-n",
    "--model-name",
    help="the name of the model"
  )

  /**
   * The file path in which to serialize the model.
   */
  val modelPath: String by parser.storing(
    "-m",
    "--model-path",
    help="the file path in which to serialize the model"
  )

  /**
   * The file path of the training dataset.
   */
  val trainingSetPath: String by parser.storing(
    "-t",
    "--training-set-path",
    help="the file path of the training dataset"
  )

  /**
   * The file path of the test dataset.
   */
  val testSetPath: String by parser.storing(
    "-s",
    "--test-set-path",
    help="the file path of the test dataset"
  )

  /**
   * The file path of the validation dataset.
   */
  val validationSetPath: String by parser.storing(
    "-v",
    "--validation-set-path",
    help="the file path of the validation dataset"
  )

  /**
   * The file path of the pre-trained word embeddings.
   */
  val embeddingsPath: String by parser.storing(
    "-e",
    "--pre-trained-word-emb-path",
    help="the file path of the pre-trained word embeddings"
  )

  /**
   * Do not optimize nor serialized the embeddings.
   */
  val noEmbeddingsOptimization: Boolean by parser.flagging(
    "--no-embeddings-optimization",
    help = "do not optimize nor serialized the embeddings"
  )

  /**
   * Auto-complete the training dataset with the missing classes.
   */
  val autoComplete: Boolean by parser.flagging(
    "-a",
    "--auto-complete",
    help = "auto-complete the training dataset with the missing classes"
  )

  /**
   * Force parsing all arguments (only read ones are parsed by default).
   */
  init {
    parser.force()
  }
}
