/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import helpers.CorpusTokenizer

/**
 * Tokenize a corpus containing plain texts, to use it as dataset for the HAN classifier.
 * The first argument is the file name of the NeuralTokenizer model, the second one of the input corpus and the third
 * one is the output file name.
 */
fun main(args: Array<String>) {

  val corpusTokenizer = CorpusTokenizer(args[0])

  corpusTokenizer.convert(args[1], args[2])
}
