/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package corpus

import com.beust.klaxon.JsonObject
import com.beust.klaxon.Klaxon
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import com.kotlinnlp.utils.getLinesCount
import java.io.File

/**
 * Convert the classes names of a one-level corpus to indices.
 *
 * The corpus must be a JSONL file with one example per line.
 * Each example is a JSON object with the following template:
 *   {"class": "CLASS_NAME", "text": "A sample text."}
 *
 * Command line arguments:
 *  1. The filename of the input corpus
 *  2. The output filename.
 */
fun main(args: Array<String>) {

  val inputFilename: String = args[0]
  val outputFilename: String = args[1]

  val inputFile = File(inputFilename)
  val outputFile = File(outputFilename)
  val jsonParser = Klaxon()
  val progress = ProgressIndicatorBar(total = getLinesCount(inputFilename))
  val categoriesMapping: MutableMap<String, Int> = mutableMapOf()

  println("Input corpus: '$inputFilename'")
  println("Output file: '$outputFilename'")

  outputFile.writer().write("") // Empty file

  inputFile.reader().forEachLine { line ->

    val example: JsonObject = jsonParser.parseJsonObject(line.reader())
    val category: String = example.string("class")!!
    val index: Int = categoriesMapping.getOrPut(category) { categoriesMapping.size }

    example["classes"] = listOf(index)

    outputFile.appendText(example.toJsonString() + "\n")

    progress.tick()
  }

  printMapping(categoriesMapping)
}

/**
 * Print the mapping of categories to indices.
 *
 * @param mapping the mapping of categories to indices
 */
private fun printMapping(mapping: Map<String, Int>) {

  println("Mapping:")

  mapping.entries.sortedBy { it.value }.forEach {
    println("%3d -> %s".format(it.value, it.key))
  }
}
