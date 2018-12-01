/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package helpers

import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonObject
import com.beust.klaxon.Parser
import com.kotlinnlp.neuraltokenizer.NeuralTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import com.kotlinnlp.neuraltokenizer.Sentence
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import java.io.*

/**
 * Tokenize corpora for the HAN classifier, converting them from the plain text format to the tokenized one.
 *
 * @param tokenizerModelFilename the filename of the model of a NeuralTokenizer
 */
class CorpusTokenizer(tokenizerModelFilename: String) {

  /**
   * The tokenizer for the texts.
   */
  private val tokenizer = NeuralTokenizer(
    model = NeuralTokenizerModel.load(FileInputStream(File(tokenizerModelFilename)))
  )

  /**
   * The JSON parser.
   */
  private val jsonParser: Parser = Parser()

  /**
   * Convert each example of a corpus from the format 1 to the format 2, tokenizing the texts.
   *   Format 1: tuple of <plain text> and <gold class>.
   *   Format 2: object with fields 'text' (as list of sentences, themselves as list of tokens) and 'class' (gold).
   *
   * @param inputFilename the filename of the input corpus (one example per line in JSON format)
   * @param outputFilename the filename in which to save the converted examples as JSON objects (one per line)
   */
  fun convert(inputFilename: String, outputFilename: String) {

    val inputFile = File(inputFilename)
    val outputFile = File(outputFilename)
    val progress = ProgressIndicatorBar(total = inputFile.reader().getLinesCount())

    println("Input corpus: '$inputFilename'")
    println("Output file: '$outputFilename'")

    outputFile.writer().write("") // Empty file

    inputFile.reader().forEachLine { line ->

      progress.tick()

      outputFile.appendText(this.convertLine(line) + "\n")
    }
  }

  /**
   * Convert a JSON line of the input file to a JSON object with 'text' and 'class' fields.
   *
   * @param line the input line
   *
   * @return the converted line as JSON string
   */
  private fun convertLine(line: String): String {

    val jsonExample: JsonObject = this.jsonParser.parse(StringBuilder(line)) as JsonObject
    val text: String = jsonExample.string("text")!!

    val tokenizedText: List<Sentence> = this.tokenizer.tokenize(text)

    jsonExample["text"] = tokenizedText.toJsonArray()

    return jsonExample.toJsonString()
  }

  /**
   * @return the number of lines of this input stream
   */
  private fun InputStreamReader.getLinesCount(): Int {

    var linesCount = 0

    this.forEachLine { linesCount++ }

    return linesCount
  }

  /**
   * @return this list of sentences converted to a nested JsonArray of token forms
   */
  private fun List<Sentence>.toJsonArray(): JsonArray<JsonArray<String>> {

    return JsonArray(*Array(
      size = this.size,
      init = { sentenceIndex ->
        JsonArray(*this[sentenceIndex].tokens.map{ it.form }.toTypedArray())
      }
    ))
  }
}
