/* Copyright 2016-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package classification

import com.xenomachina.argparser.ArgParser
import com.xenomachina.argparser.default

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
   * The file path of the HAN classifier serialized model.
   */
  val classifierModelPath: String by parser.storing(
    "-m",
    "--classifier-model-path",
    help="the file path of the HAN classifier serialized model"
  )

  /**
   * The file path of the tokenizer serialized model.
   */
  val tokenizerModelPath: String by parser.storing(
    "-t",
    "--tokenizer-model-path",
    help="the file path of the tokenizer serialized model"
  )

  /**
   * Reduce the size of the sentences to use for the classification.
   */
  val reduceSentences: Boolean by parser.flagging(
    "-r",
    "--reduce-sentences",
    help="reduce the size of the sentences to use for the classification"
  ).default { false }

  /**
   * The file path of the labels configuration.
   */
  val labelsConfigPath: String? by parser.storing(
    "-l",
    "--labels-config-path",
    help="the file path of the labels configuration"
  ).default { null }

  /**
   * Force parsing all arguments (only read ones are parsed by default).
   */
  init {
    parser.force()
  }
}
