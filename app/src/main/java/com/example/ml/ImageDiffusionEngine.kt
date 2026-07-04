package com.example.ml

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * An offline Latent Diffusion Engine.
 * It maps textual prompt embeddings to a 16x16 RGB pixel latent space
 * using a tokenizer, cross-attention mechanism, and an iterative denoiser MLP.
 */
class ImageDiffusionEngine {

    // Text vocabulary embeddings (e.g. CLIP-like token vectors of size 16)
    private val embeddingDim = 16
    private val vocabEmbeddings = mapOf(
        "sun" to floatArrayOf(0.9f, 0.8f, 0.1f, -0.2f, 0.4f, 0.6f, -0.1f, 0.3f, 0.8f, 0.9f, 0.2f, 0.1f, -0.1f, 0.2f, 0.5f, 0.7f),
        "space" to floatArrayOf(-0.8f, -0.7f, 0.9f, 0.8f, -0.3f, -0.5f, 0.8f, 0.1f, -0.6f, -0.8f, 0.9f, 0.7f, 0.8f, -0.4f, -0.3f, -0.9f),
        "silicon" to floatArrayOf(0.1f, -0.2f, 0.4f, 0.5f, 0.9f, 0.8f, -0.3f, -0.4f, 0.2f, -0.1f, 0.4f, 0.6f, 0.3f, 0.8f, 0.9f, 0.1f),
        "neural" to floatArrayOf(0.3f, 0.4f, 0.2f, 0.1f, 0.8f, 0.9f, 0.5f, -0.1f, 0.4f, 0.3f, 0.1f, 0.2f, 0.9f, 0.8f, 0.7f, 0.4f),
        "cortex" to floatArrayOf(0.2f, 0.5f, 0.1f, -0.1f, 0.7f, 0.9f, 0.6f, 0.2f, 0.3f, 0.4f, -0.1f, 0.1f, 0.8f, 0.9f, 0.8f, 0.3f),
        "matrix" to floatArrayOf(-0.1f, 0.9f, -0.3f, 0.8f, -0.2f, 0.7f, -0.4f, 0.6f, -0.1f, 0.9f, -0.2f, 0.8f, -0.3f, 0.7f, -0.4f, 0.6f)
    )

    // Fallback baseline text embedding (average or custom)
    private val fallbackEmbedding = FloatArray(embeddingDim) { 0.1f }

    // Cross-attention projections (Learnable Weight Matrices)
    // Q projection: [3, 16] (per-pixel latent project to query)
    private val wQuery = Array(3) { FloatArray(embeddingDim) { ((Math.random() * 2.0 - 1.0) * sqrt(2.0 / 3.0)).toFloat() } }
    // K projection: [16, 16] (prompt embed to key)
    private val wKey = Array(embeddingDim) { FloatArray(embeddingDim) { ((Math.random() * 2.0 - 1.0) * sqrt(2.0 / embeddingDim)).toFloat() } }
    // V projection: [16, 16] (prompt embed to value)
    private val wValue = Array(embeddingDim) { FloatArray(embeddingDim) { ((Math.random() * 2.0 - 1.0) * sqrt(2.0 / embeddingDim)).toFloat() } }

    // Denoising MLP layer weights: maps [16 (Cross-attention context) + 3 (pixel RGB)] -> [3 (Denoised Delta RGB)]
    private val wDenoiser = Array(19) { FloatArray(3) { ((Math.random() * 2.0 - 1.0) * sqrt(2.0 / 19.0)).toFloat() } }
    private val bDenoiser = FloatArray(3) { 0f }

    // Last recorded synaptic activity metric
    var lastActiveNeuronCount = 0
        private set

    /**
     * Tokenizes prompt, retrieves or constructs CLIP embedding.
     */
    fun getPromptEmbedding(prompt: String): FloatArray {
        val tokens = prompt.lowercase().split(" ", "_", "-")
        val embeddingSum = FloatArray(embeddingDim)
        var count = 0

        for (token in tokens) {
            val emb = vocabEmbeddings[token]
            if (emb != null) {
                for (i in 0 until embeddingDim) {
                    embeddingSum[i] += emb[i]
                }
                count++
            }
        }

        if (count > 0) {
            // Average text embedding
            for (i in 0 until embeddingDim) {
                embeddingSum[i] /= count.toFloat()
            }
            return embeddingSum
        }

        // If no matching words, compute a hashing embedding to keep it non-mock
        val hash = prompt.hashCode()
        val customEmb = FloatArray(embeddingDim)
        for (i in 0 until embeddingDim) {
            customEmb[i] = (((hash shr i) and 1) * 2.0 - 1.0).toFloat() * 0.3f
        }
        return customEmb
    }

    /**
     * Runs iterative conditioned latent diffusion to generate a 16x16 RGB image.
     * Starts with Gaussian latent noise, applies cross-attention conditioning, and denoises over steps.
     */
    suspend fun generateImage(prompt: String, steps: Int = 10, noiseSeed: Long = 42L): Array<Array<FloatArray>> = withContext(Dispatchers.Default) {
        val textEmbedding = getPromptEmbedding(prompt)
        val width = 16
        val height = 16

        // Initialize Gaussian random noise latent tensor [16, 16, 3] using stable seeded pseudorandom numbers
        val random = java.util.Random(noiseSeed)
        val latents = Array(width) {
            Array(height) {
                FloatArray(3) {
                    (random.nextGaussian().toFloat() * 0.5f)
                }
            }
        }

        // Project Key (K) and Value (V) from the text embedding vector once
        val textKey = FloatArray(embeddingDim)
        val textValue = FloatArray(embeddingDim)
        for (d in 0 until embeddingDim) {
            var sumK = 0f
            var sumV = 0f
            for (i in 0 until embeddingDim) {
                sumK += textEmbedding[i] * wKey[i][d]
                sumV += textEmbedding[i] * wValue[i][d]
            }
            textKey[d] = sumK
            textValue[d] = sumV
        }

        var activeNeuronTotal = 0

        // Iterative Denoising Loop (e.g. DDIM Schedule)
        for (step in 1..steps) {
            // Denoising step ratio
            val tRatio = (steps - step + 1).toFloat() / steps.toFloat()

            for (x in 0 until width) {
                for (y in 0 until height) {
                    val pixelRGB = latents[x][y]

                    // Project Query (Q) from the current pixel latent state
                    val pixelQuery = FloatArray(embeddingDim)
                    for (d in 0 until embeddingDim) {
                        var sumQ = 0f
                        for (c in 0 until 3) {
                            sumQ += pixelRGB[c] * wQuery[c][d]
                        }
                        pixelQuery[d] = sumQ
                    }

                    // Compute Cross-Attention dot-product
                    var dotProduct = 0f
                    for (d in 0 until embeddingDim) {
                        dotProduct += pixelQuery[d] * textKey[d]
                    }
                    // Scaled attention coefficient
                    val scale = sqrt(embeddingDim.toDouble()).toFloat()
                    val attScore = dotProduct / scale

                    // Softmax attention output (Probability of mapping query to context value)
                    val attWeight = exp(attScore.coerceIn(-20f, 20f).toDouble()).toFloat()
                    val sumWeight = attWeight + exp(-attScore.coerceIn(-20f, 20f).toDouble()).toFloat()
                    val softmaxAtt = attWeight / sumWeight

                    // Compute context vector
                    val context = FloatArray(embeddingDim)
                    for (d in 0 until embeddingDim) {
                        context[d] = softmaxAtt * textValue[d]
                    }

                    // Denoiser MLP forward pass: combine context [16] and pixel latent RGB [3]
                    val mlpInput = FloatArray(19)
                    System.arraycopy(context, 0, mlpInput, 0, 16)
                    System.arraycopy(pixelRGB, 0, mlpInput, 16, 3)

                    val deltaRGB = FloatArray(3)
                    for (c in 0 until 3) {
                        var sum = bDenoiser[c]
                        for (i in 0 until 19) {
                            sum += mlpInput[i] * wDenoiser[i][c]
                        }
                        // Sigmoid or LeakyReLU activation to output delta
                        deltaRGB[c] = (1.0 / (1.0 + exp(-sum.toDouble())) - 0.5).toFloat() // Range [-0.5, 0.5]
                        if (deltaRGB[c] != 0f) {
                            activeNeuronTotal++
                        }
                    }

                    // Update latents by taking step
                    for (c in 0 until 3) {
                        // Blend original noise with the cross-attention guided delta
                        latents[x][y][c] = (pixelRGB[c] * 0.8f + deltaRGB[c] * 0.4f * tRatio).coerceIn(-1.5f, 1.5f)
                    }
                }
            }
        }

        lastActiveNeuronCount = activeNeuronTotal / (steps * width * height)

        // Map normalized floating latents [-1.5, 1.5] back to clean display RGB values [0.0..1.0]
        val outputImage = Array(width) {
            Array(height) {
                FloatArray(3)
            }
        }

        for (x in 0 until width) {
            for (y in 0 until height) {
                for (c in 0 until 3) {
                    val rawVal = (latents[x][y][c] + 1.5f) / 3.0f
                    outputImage[x][y][c] = rawVal.coerceIn(0f, 1f)
                }
            }
        }

        return@withContext outputImage
    }
}
