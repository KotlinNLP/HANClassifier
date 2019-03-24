/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.hanclassifier

import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.deeplearning.attention.han.*
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * A classifier based on Hierarchic Attention Networks.
 *
 * @param han a HAN model
 * @param useDropout whether to apply the dropout during the forward (default = false)
 * @param propagateToInput whether to propagate the errors to the input during the backward (default = false)
 */
internal class HANClassifierSingle(
  val han: HAN,
  override val useDropout: Boolean = false,
  override val propagateToInput: Boolean = false,
  override val id: Int = 0
) : NeuralProcessor<
  List<EncodedSentence>, // InputType
  DenseNDArray, // OutputType
  DenseNDArray, // ErrorsType
  List<EncodedSentence> // InputErrorsType
  > {

  /**
   * The [HANEncoder] used as classifier (Softmax output activation).
   */
  private val encoder = HANEncoder<DenseNDArray>(
    model = this.han,
    useDropout = this.useDropout,
    propagateToInput = this.propagateToInput)

  /**
   * Classify the given [input].
   *
   * @param input a list of encoded sentences
   *
   * @return the probability distribution of the classification
   */
  override fun forward(input: List<EncodedSentence>): DenseNDArray = this.encoder.forward(input.toHierarchyGroup())

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
  override fun getInputErrors(copy: Boolean): List<EncodedSentence> =
    (this.encoder.getInputErrors(copy = false) as HierarchyGroup).map { group ->
      EncodedSentence(tokens = (group as HierarchySequence<*>).map { it as DenseNDArray })
    }

  /**
   * Return the params errors of the last backward.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference (default true)
   *
   * @return the parameters errors
   */
  override fun getParamsErrors(copy: Boolean): ParamsErrorsList = this.encoder.getParamsErrors(copy)

  /**
   * @return the hierarchy group built from this list of sentences
   */
  private fun List<EncodedSentence>.toHierarchyGroup(): HierarchyGroup =
    HierarchyGroup(*this.map { it.toHierarchySequence() }.toTypedArray())

  /**
   * @return the hierarchy sequence built from sentence
   */
  private fun EncodedSentence.toHierarchySequence(): HierarchySequence<DenseNDArray> =
    HierarchySequence(*this.tokens.map {it }.toTypedArray())
}
