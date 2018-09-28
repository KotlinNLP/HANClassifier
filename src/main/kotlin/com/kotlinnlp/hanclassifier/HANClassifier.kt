/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.attention.han.HANEncoder
import com.kotlinnlp.simplednn.deeplearning.attention.han.HANParameters
import com.kotlinnlp.simplednn.deeplearning.attention.han.HierarchyGroup
import com.kotlinnlp.simplednn.deeplearning.attention.han.HierarchySequence
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A classifier based on Hierarchic Attention Networks.
 *
 * @param model the model of this [HANClassifier]
 * @param useDropout whether to apply the dropout during the forward
 * @param propagateToInput whether to propagate the errors to the input during the backward
 */
class HANClassifier(
  val model: HANClassifierModel,
  override val useDropout: Boolean,
  override val propagateToInput: Boolean,
  override val id: Int = 0
) : NeuralProcessor<
  List<Sentence<FormToken>>, // InputType
  DenseNDArray, // OutputType
  DenseNDArray, // ErrorsType
  HierarchyGroup, // InputErrorsType
  HANParameters // ParamsType
  > {

  /**
   * The [HANEncoder] used as classifier (Softmax output activation).
   */
  private val encoder = HANEncoder<DenseNDArray>(
    model = this.model.han,
    useDropout = this.useDropout,
    propagateToInput = this.propagateToInput)

  /**
   * Classify the given [input].
   *
   * @param input a list of sentences of form tokens
   *
   * @return the probability distribution of the classification
   */
  override fun forward(input: List<Sentence<FormToken>>): DenseNDArray =
    this.encoder.forward(input.toHierarchyGroup(this.model.embeddings))

  /**
   * The Backward.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: DenseNDArray) = this.encoder.backward(outputErrors)

  /**
   * Return the input errors of the last backward.
   * Before calling this method make sure that [propagateToInput] is enabled.
   *
   * @param copy whether to return by value or by reference (default true)
   *
   * @return the input errors
   */
  override fun getInputErrors(copy: Boolean): HierarchyGroup =
    this.encoder.getInputErrors(copy = false) as HierarchyGroup

  /**
   * Return the params errors of the last backward.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference (default true)
   *
   * @return the parameters errors
   */
  override fun getParamsErrors(copy: Boolean): HANParameters = this.encoder.getParamsErrors(copy)

  /**
   * @param embeddings an embeddings map of forms
   *
   * @return the hierarchy group built from this list of sentences
   */
  private fun List<Sentence<FormToken>>.toHierarchyGroup(embeddings: EmbeddingsMap<String>): HierarchyGroup =
    HierarchyGroup(*this.map { it.toHierarchySequence(embeddings) }.toTypedArray())

  /**
   * @param embeddings an embeddings map of forms
   *
   * @return the hierarchy sequence built from sentence
   */
  private fun Sentence<FormToken>.toHierarchySequence(
    embeddings: EmbeddingsMap<String>
  ): HierarchySequence<DenseNDArray> = HierarchySequence(
    *this.tokens.map { embeddings.get(key = it.form).array.values }.toTypedArray()
  )
}
