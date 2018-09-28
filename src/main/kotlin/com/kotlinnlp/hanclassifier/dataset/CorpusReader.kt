/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier.dataset

import com.beust.klaxon.*
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.linguisticdescription.sentence.Sentence as SentenceInterface
import java.io.File
import java.lang.StringBuilder

/**
 * The corpus reader.
 */
object CorpusReader {

  /**
   * A sentence token.
   *
   * @property form the form of the token
   */
  class Token(override val form: String) : FormToken

  /**
   * An input sentence.
   *
   * @property tokens the list of tokens that compose this sentence
   */
  class Sentence(override val tokens: List<Token>) : SentenceInterface<FormToken>

  /**
   * Read examples from a given JSON file.
   *
   * @param filename the name of the file containing the dataset in JSON format
   *
   * @return a list of examples
   */
  fun read(filename: String): List<Example> {

    val examples = mutableListOf<Example>()
    val parser = Parser()

    File(filename).reader().forEachLine { line ->

      val parsedExample = parser.parse(StringBuilder(line)) as JsonObject
      val sentences = parsedExample.array<JsonArray<String>>("text")!!

      examples.add(Example(
        inputText = sentences.map { forms -> Sentence(tokens = forms.map { Token(form = it) }) },
        outputGold = parsedExample.int("class")!! - 1
      ))
    }

    return examples
  }
}
