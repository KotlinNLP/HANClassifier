/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier.dataset

import com.beust.klaxon.*
import java.io.ByteArrayInputStream
import java.io.File
import java.io.InputStream

/**
 *
 */
object CorpusReader {

  /**
   *
   */
  fun read(filename: String): ArrayList<Example> {

    val examples = arrayListOf<Example>()
    val parser = Parser()

    File(filename).reader().forEachLine { line ->

      val parsedExample = parser.parse(line.toInputStream()) as JsonObject
      val sentences = parsedExample.array<JsonArray<String>>("text")!!

      examples.add(Example(
        inputText = sentences.toKotlinList(),
        outputGold = parsedExample.int("class")!! - 1
      ))
    }

    return examples
  }

  /**
   *
   */
  private fun String.toInputStream(): InputStream = ByteArrayInputStream(this.toByteArray())

  /**
   *
   */
  private fun JsonArray<JsonArray<String>>.toKotlinList(): List<List<String>> = List(
    size = this.size,
    init = { i -> this[i].toList() }
  )
}
