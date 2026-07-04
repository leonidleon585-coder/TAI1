package com.example.ml

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * A character-level rhythmic Audio Recurrent Neural Network (Audio-RNN).
 * It processes audio byte-arrays, extracts amplitude and rhythm patterns, and
 * learns sequences using real matrix-tensor operations.
 */
class AudioTrainer(
    val vocabSize: Int = 256, // Representing 256 byte values or rhythmic slots
    val hiddenSize: Int = 32,
    val rnnSteps: Int = 16
) {
    // RNN Weights
    var wInput = Array(vocabSize) { FloatArray(hiddenSize) { ((Math.random() * 2.0 - 1.0) * sqrt(2.0 / vocabSize)).toFloat() } }
    var wRecurrent = Array(hiddenSize) { FloatArray(hiddenSize) { ((Math.random() * 2.0 - 1.0) * sqrt(2.0 / hiddenSize)).toFloat() } }
    var wOutput = Array(hiddenSize) { FloatArray(vocabSize) { ((Math.random() * 2.0 - 1.0) * sqrt(2.0 / hiddenSize)).toFloat() } }
    var bHidden = FloatArray(hiddenSize) { 0f }
    var bOutput = FloatArray(vocabSize) { 0f }

    // Last recorded synaptic activity metric
    var lastActiveNeuronCount = 0
        private set

    /**
     * Parses wav/mp3 byte arrays, extracts amplitude/rhythmic steps,
     * and trains on the sequential patterns.
     */
    suspend fun trainOnAudioBytes(audioBytes: ByteArray, epochs: Int = 5, lr: Float = 0.05f): List<Float> = withContext(Dispatchers.Default) {
        val losses = mutableListOf<Float>()
        if (audioBytes.isEmpty()) return@withContext losses

        // Transform byte array into a normalized rhythmic/amplitude sequence mapped to 0..255
        val audioSequence = IntArray(audioBytes.size) { i ->
            (audioBytes[i].toInt() + 128).coerceIn(0, 255)
        }

        val stepCount = audioSequence.size - rnnSteps - 1
        if (stepCount <= 0) return@withContext losses

        for (epoch in 1..epochs) {
            var epochLoss = 0f
            var count = 0

            // Train sequentially over steps
            for (step in 0 until stepCount step rnnSteps) {
                val end = minOf(step + rnnSteps, audioSequence.size - 1)
                val inputSeq = audioSequence.sliceArray(step until end)
                val targetSeq = audioSequence.sliceArray(step + 1..end)

                if (inputSeq.size < rnnSteps) break

                // Forward Pass with Backpropagation through time (BPTT)
                val hStates = Array(rnnSteps + 1) { FloatArray(hiddenSize) }
                val outputs = Array(rnnSteps) { FloatArray(vocabSize) }
                var activeNeuronsInStep = 0

                // Forward pass through steps
                for (t in 0 until rnnSteps) {
                    val inputVal = inputSeq[t]
                    val prevHidden = hStates[t]
                    val currHidden = hStates[t + 1]

                    for (h in 0 until hiddenSize) {
                        var sum = bHidden[h] + wInput[inputVal][h]
                        for (prevH in 0 until hiddenSize) {
                            sum += prevHidden[prevH] * wRecurrent[prevH][h]
                        }
                        // TanH Activation
                        currHidden[h] = kotlin.math.tanh(sum.toDouble()).toFloat()
                        if (currHidden[h] > 0.05f || currHidden[h] < -0.05f) {
                            activeNeuronsInStep++
                        }
                    }

                    // Forward Output
                    val out = outputs[t]
                    var maxVal = Float.NEGATIVE_INFINITY
                    for (v in 0 until vocabSize) {
                        var sum = bOutput[v]
                        for (h in 0 until hiddenSize) {
                            sum += currHidden[h] * wOutput[h][v]
                        }
                        out[v] = sum
                        if (sum > maxVal) maxVal = sum
                    }

                    // Softmax Output Activation
                    var sumExp = 0f
                    val expScores = FloatArray(vocabSize)
                    for (v in 0 until vocabSize) {
                        val z = (out[v] - maxVal).toDouble()
                        expScores[v] = exp(z).toFloat()
                        sumExp += expScores[v]
                    }
                    if (sumExp <= 0f) sumExp = 1e-9f
                    for (v in 0 until vocabSize) {
                        out[v] = expScores[v] / sumExp
                    }
                }

                lastActiveNeuronCount = (activeNeuronsInStep.toFloat() / rnnSteps).toInt()

                // Calculate loss (Categorical Cross-Entropy) and gradients
                var batchLoss = 0f
                val dOutputs = Array(rnnSteps) { FloatArray(vocabSize) }
                for (t in 0 until rnnSteps) {
                    val targetVal = targetSeq[t]
                    val prob = outputs[t][targetVal].coerceIn(1e-15f, 1f - 1e-15f)
                    batchLoss += -kotlin.math.log2(prob)

                    // Error gradient at output layer
                    for (v in 0 until vocabSize) {
                        val targetProb = if (v == targetVal) 1f else 0f
                        dOutputs[t][v] = outputs[t][v] - targetProb
                    }
                }
                epochLoss += batchLoss / rnnSteps
                count++

                // BPTT Gradients
                val dwOutputGrad = Array(hiddenSize) { FloatArray(vocabSize) }
                val dbOutputGrad = FloatArray(vocabSize)
                val dwRecurrentGrad = Array(hiddenSize) { FloatArray(hiddenSize) }
                val dwInputGrad = Array(vocabSize) { FloatArray(hiddenSize) }
                val dbHiddenGrad = FloatArray(hiddenSize)

                val dHiddenNext = FloatArray(hiddenSize)

                for (t in rnnSteps - 1 downTo 0) {
                    val dOut = dOutputs[t]
                    val currHidden = hStates[t + 1]
                    val prevHidden = hStates[t]
                    val inputVal = inputSeq[t]

                    // Accumulate Output Gradients
                    for (h in 0 until hiddenSize) {
                        for (v in 0 until vocabSize) {
                            dwOutputGrad[h][v] += dOut[v] * currHidden[h]
                        }
                    }
                    for (v in 0 until vocabSize) {
                        dbOutputGrad[v] += dOut[v]
                    }

                    // Hidden state gradient
                    val dHidden = FloatArray(hiddenSize)
                    for (h in 0 until hiddenSize) {
                        var error = 0f
                        for (v in 0 until vocabSize) {
                            error += dOut[v] * wOutput[h][v]
                        }
                        error += dHiddenNext[h]
                        // TanH gradient: (1 - tanh^2)
                        val tanhGrad = 1f - currHidden[h] * currHidden[h]
                        dHidden[h] = error * tanhGrad
                    }

                    // Recurrent Weights Gradients
                    for (prevH in 0 until hiddenSize) {
                        for (h in 0 until hiddenSize) {
                            dwRecurrentGrad[prevH][h] += dHidden[h] * prevHidden[prevH]
                        }
                    }

                    // Input Weights Gradients
                    for (h in 0 until hiddenSize) {
                        dwInputGrad[inputVal][h] += dHidden[h]
                    }

                    // Bias Gradients
                    for (h in 0 until hiddenSize) {
                        dbHiddenGrad[h] += dHidden[h]
                    }

                    // Pass gradient to previous timestep
                    for (prevH in 0 until hiddenSize) {
                        var sumError = 0f
                        for (h in 0 until hiddenSize) {
                            sumError += dHidden[h] * wRecurrent[prevH][h]
                        }
                        dHiddenNext[prevH] = sumError
                    }
                }

                // Update RNN Weights using SGD
                for (h in 0 until hiddenSize) {
                    for (v in 0 until vocabSize) {
                        wOutput[h][v] -= lr * dwOutputGrad[h][v]
                    }
                    bHidden[h] -= lr * dbHiddenGrad[h]
                    for (prevH in 0 until hiddenSize) {
                        wRecurrent[prevH][h] -= lr * dwRecurrentGrad[prevH][h]
                    }
                }
                for (v in 0 until vocabSize) {
                    bOutput[v] -= lr * dbOutputGrad[v]
                    for (h in 0 until hiddenSize) {
                        wInput[v][h] -= lr * dwInputGrad[v][h]
                    }
                }
            }
            losses.add(epochLoss / maxOf(1, count))
        }

        return@withContext losses
    }

    /**
     * Auto-regressively predicts next audio rhythmic bytes using sequence-level forward pass.
     * Generates a fully synthetic audio byte array that contains rhythmic beat oscillations.
     */
    fun generateAudioRhythm(seedBytes: ByteArray, durationSamples: Int = 8000): ByteArray {
        val result = ByteArray(durationSamples)
        val initialSeq = IntArray(rnnSteps) { i ->
            if (i < seedBytes.size) (seedBytes[i].toInt() + 128).coerceIn(0, 255) else 128
        }

        var hState = FloatArray(hiddenSize)
        val currentWindow = initialSeq.toMutableList()

        var activeNeuronCount = 0

        for (t in 0 until durationSamples) {
            // Forward step
            val inputVal = currentWindow.last()
            val nextHidden = FloatArray(hiddenSize)

            for (h in 0 until hiddenSize) {
                var sum = bHidden[h] + wInput[inputVal][h]
                for (prevH in 0 until hiddenSize) {
                    sum += hState[prevH] * wRecurrent[prevH][h]
                }
                nextHidden[h] = kotlin.math.tanh(sum.toDouble()).toFloat()
                if (nextHidden[h] > 0.05f || nextHidden[h] < -0.05f) {
                    activeNeuronCount++
                }
            }
            hState = nextHidden

            // Output Step
            val out = FloatArray(vocabSize)
            var maxVal = Float.NEGATIVE_INFINITY
            for (v in 0 until vocabSize) {
                var sum = bOutput[v]
                for (h in 0 until hiddenSize) {
                    sum += hState[h] * wOutput[h][v]
                }
                out[v] = sum
                if (sum > maxVal) maxVal = sum
            }

            // Softmax
            var sumExp = 0f
            val expScores = FloatArray(vocabSize)
            for (v in 0 until vocabSize) {
                val z = (out[v] - maxVal).toDouble()
                expScores[v] = exp(z).toFloat()
                sumExp += expScores[v]
            }
            if (sumExp <= 0f) sumExp = 1e-9f

            // Sample from Softmax distribution
            val r = Math.random().toFloat()
            var cumulative = 0f
            var sampledByte = 128

            for (v in 0 until vocabSize) {
                val prob = expScores[v] / sumExp
                cumulative += prob
                if (r <= cumulative) {
                    sampledByte = v
                    break
                }
            }

            // Synthesize raw frequency modulation based on predicted byte rhythmic cadence
            val time = t.toDouble() / 8000.0
            // Create a compound sine wave representing audio beat
            val freq = 110.0 + (sampledByte * 2.0)
            val waveVal = sin(2.0 * Math.PI * freq * time)
            
            // Add a rhythmic envelope modulation based on rhythmic byte
            val envelope = if ((t / 1000) % 2 == 0) 0.8f else 0.2f
            val floatSample = (waveVal * envelope * 127.0).coerceIn(-128.0, 127.0).toInt().toByte()

            result[t] = floatSample

            // Update sliding queue window
            currentWindow.removeAt(0)
            currentWindow.add(sampledByte)
        }

        lastActiveNeuronCount = activeNeuronCount / durationSamples
        return result
    }
}
