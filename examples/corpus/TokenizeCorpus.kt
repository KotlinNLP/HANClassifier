/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package corpus

/**
 * Tokenize a corpus containing plain texts, to use it as dataset for the HAN classifier.
 *
 * Command line arguments:
 *  1. The filename of the NeuralTokenizer model
 *  2. The filename of the input corpus
 *  3. The output filename.
 */
fun main(args: Array<String>) {

  val corpusTokenizer = CorpusTokenizer(args[0])

  corpusTokenizer.convert(args[1], args[2])
}
