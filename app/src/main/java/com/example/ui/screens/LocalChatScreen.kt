package com.example.ui.screens

import android.widget.Toast
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.ml.TrainerEngine
import com.example.ml.TrainingState
import com.example.ui.CyberCard
import com.example.ui.LayoutConstants
import com.example.ui.ScreenHeader
import com.example.ui.SynapticActivityGraph
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

enum class ConsoleMode(val label: String, val icon: androidx.compose.ui.graphics.vector.ImageVector) {
    TEXT("Text LLM", Icons.Default.Chat),
    AUDIO("Audio RNN", Icons.Default.Audiotrack),
    IMAGE("Latent Diffusion", Icons.Default.Image)
}

data class LocalChatBubble(
    val id: String,
    val sender: String, // "User", "Offline Model", or "System"
    val text: String,
    val promptSeed: String = "", // Seed context used for feedback corrections
    val timestamp: Long = System.currentTimeMillis(),
    val isStreaming: Boolean = false,
    val audioWave: ByteArray? = null, // Audio synthesis output
    val diffuseImage: Array<Array<FloatArray>>? = null // Conditioned image generation output
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LocalChatScreen(
    trainerEngine: TrainerEngine,
    state: TrainingState,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val scrollState = rememberLazyListState()

    var activeMode by remember { mutableStateOf(ConsoleMode.TEXT) }
    var textInput by remember { mutableStateOf("") }
    var isGenerating by remember { mutableStateOf(false) }

    // Conversational state including generated visual outputs
    var chatMessages by remember {
        mutableStateOf(
            listOf(
                LocalChatBubble(
                    id = "welcome",
                    sender = "Offline Model",
                    text = "System cortex initialized. I am your fully autonomous Next-Character Generative Network running directly on this Snapdragon chipset. Enter a prompt or select a multi-modal core."
                )
            )
        )
    }

    // Auto-scroll to the bottom of conversation
    LaunchedEffect(chatMessages.size) {
        if (chatMessages.isNotEmpty()) {
            scrollState.animateScrollToItem(chatMessages.size - 1)
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(LayoutConstants.ColorDarkBg)
    ) {
        // Multi-modal Console Selectors (Adaptive Grid 8dp)
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            ConsoleMode.values().forEach { mode ->
                val selected = activeMode == mode
                FilterChip(
                    selected = selected,
                    onClick = { activeMode = mode },
                    label = { 
                        Text(
                            text = mode.label, 
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold,
                            fontSize = 11.sp
                        ) 
                    },
                    leadingIcon = {
                        Icon(
                            imageVector = mode.icon,
                            contentDescription = mode.label,
                            modifier = Modifier.size(16.dp)
                        )
                    },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = LayoutConstants.ColorCyberGreen,
                        selectedLabelColor = Color.Black,
                        selectedLeadingIconColor = Color.Black,
                        containerColor = LayoutConstants.ColorDarkSurface,
                        labelColor = Color.LightGray,
                        iconColor = Color.LightGray
                    ),
                    modifier = Modifier.weight(1f)
                )
            }
        }

        // Display Live Synaptic Activity Graph
        SynapticActivityGraph(
            history = state.activeNeuronsHistory,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)
        )

        // Conversational Feed
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 4.dp)
        ) {
            LazyColumn(
                state = scrollState,
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                items(chatMessages, key = { it.id }) { bubble ->
                    MultiModalBubbleRow(
                        bubble = bubble,
                        onCorrectionSubmit = { contextSeed, expectedChar ->
                            val currentLoss = trainerEngine.applyCorrectionFeedback(contextSeed, expectedChar)
                            Toast.makeText(
                                context,
                                "Reinforcement gradient backpropagated. Loss drops to: ${String.format("%.4f", currentLoss)}",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    )
                }
            }

            if (chatMessages.size == 1 && activeMode != ConsoleMode.TEXT) {
                Text(
                    text = "Use the console below to input seed terms and generate multi-modal outputs. Synaptic metrics are fully offline.",
                    style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center,
                    color = Color.DarkGray,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier
                        .align(Alignment.Center)
                        .padding(32.dp)
                )
            }
        }

        // Input Console Pane
        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .navigationBarsPadding(),
            color = LayoutConstants.ColorDarkSurface,
            tonalElevation = 4.dp,
            border = BorderStroke(1.dp, LayoutConstants.ColorBorder)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = textInput,
                    onValueChange = { textInput = it },
                    placeholder = { 
                        Text(
                            text = when(activeMode) {
                                ConsoleMode.TEXT -> "Feed a seed word to generate local responses..."
                                ConsoleMode.AUDIO -> "Enter seed character sequence to trigger rhythm..."
                                ConsoleMode.IMAGE -> "Condition text (Sun, Space, Silicon, Matrix)..."
                            }, 
                            color = Color.Gray, 
                            style = MaterialTheme.typography.bodyMedium
                        ) 
                    },
                    singleLine = true,
                    enabled = !isGenerating,
                    modifier = Modifier
                        .weight(1f)
                        .height(54.dp),
                    shape = RoundedCornerShape(8.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = LayoutConstants.ColorCyberGreen,
                        unfocusedBorderColor = Color(0xFF334155),
                        focusedLabelColor = LayoutConstants.ColorCyberGreen
                    )
                )

                Spacer(modifier = Modifier.width(8.dp))

                Button(
                    onClick = {
                        if (textInput.trim().isEmpty()) return@Button
                        
                        val prompt = textInput
                        val userMessageId = "user_${System.currentTimeMillis()}"
                        val modelMessageId = "model_${System.currentTimeMillis()}"
                        
                        // Add User message
                        chatMessages = chatMessages + LocalChatBubble(
                            id = userMessageId,
                            sender = "User",
                            text = prompt
                        )
                        textInput = ""
                        isGenerating = true

                        scope.launch {
                            when(activeMode) {
                                ConsoleMode.TEXT -> {
                                    // Auto-regressive character sequence prediction
                                    val fullResponse = trainerEngine.generateText(prompt, length = 100)
                                    
                                    chatMessages = chatMessages + LocalChatBubble(
                                        id = modelMessageId,
                                        sender = "Offline Model",
                                        text = "",
                                        promptSeed = prompt,
                                        isStreaming = true
                                    )

                                    var currentStreamedText = ""
                                    for (char in fullResponse) {
                                        currentStreamedText += char
                                        chatMessages = chatMessages.map { bubble ->
                                            if (bubble.id == modelMessageId) {
                                                bubble.copy(text = currentStreamedText)
                                            } else {
                                                bubble
                                            }
                                        }
                                        delay(15) // Typewriter cadence
                                    }

                                    chatMessages = chatMessages.map { bubble ->
                                        if (bubble.id == modelMessageId) {
                                            bubble.copy(isStreaming = false)
                                        } else {
                                            bubble
                                        }
                                    }
                                }

                                ConsoleMode.AUDIO -> {
                                    // Rhythm prediction on byte streams
                                    val seedBytes = prompt.toByteArray(Charsets.UTF_8)
                                    val synthesisBytes = trainerEngine.audioTrainer.generateAudioRhythm(seedBytes, durationSamples = 3000)
                                    
                                    // Push real-time synaptic spike to state
                                    val currentSpikes = state.activeNeuronsHistory.toMutableList()
                                    currentSpikes.add(trainerEngine.audioTrainer.lastActiveNeuronCount)
                                    if (currentSpikes.size > 40) currentSpikes.removeAt(0)
                                    trainerEngine.updateState { it.copy(activeNeuronsHistory = currentSpikes) }

                                    chatMessages = chatMessages + LocalChatBubble(
                                        id = modelMessageId,
                                        sender = "Offline Model",
                                        text = "Synthetic Audio Rhythm prediction complete. Generated 3,000 sound byte arrays utilizing character recurrent loops: ${synthesisBytes.size} rhythmic samples compiled.",
                                        audioWave = synthesisBytes
                                    )
                                }

                                ConsoleMode.IMAGE -> {
                                    // Conditioned Latent Diffusion
                                    val imagePixels = trainerEngine.imageDiffusionEngine.generateImage(prompt, steps = 10)
                                    
                                    val currentSpikes = state.activeNeuronsHistory.toMutableList()
                                    currentSpikes.add(trainerEngine.imageDiffusionEngine.lastActiveNeuronCount)
                                    if (currentSpikes.size > 40) currentSpikes.removeAt(0)
                                    trainerEngine.updateState { it.copy(activeNeuronsHistory = currentSpikes) }

                                    chatMessages = chatMessages + LocalChatBubble(
                                        id = modelMessageId,
                                        sender = "Offline Model",
                                        text = "Latent Diffusion completed in-memory. Cross-attention operation successfully mapped textual embedding condition '$prompt' to a 16x16 RGB latent noise space.",
                                        diffuseImage = imagePixels
                                    )
                                }
                            }
                            isGenerating = false
                        }
                    },
                    enabled = textInput.trim().isNotEmpty() && !isGenerating,
                    modifier = Modifier.height(54.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = LayoutConstants.ColorCyberGreen),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.Send,
                        contentDescription = "Send prompt",
                        tint = Color.Black,
                        modifier = Modifier.size(18.dp)
                    )
                }
            }
        }
    }
}

@Composable
fun MultiModalBubbleRow(
    bubble: LocalChatBubble,
    onCorrectionSubmit: (contextSeed: String, expectedChar: Char) -> Unit
) {
    val isUser = bubble.sender == "User"
    var showCorrectionInput by remember { mutableStateOf(false) }
    var correctiveCharInput by remember { mutableStateOf("") }

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
        verticalAlignment = Alignment.Top
    ) {
        if (!isUser) {
            Box(
                modifier = Modifier
                    .size(36.dp)
                    .clip(RoundedCornerShape(18.dp))
                    .background(LayoutConstants.ColorCyberGreen.copy(alpha = 0.2f))
                    .border(1.dp, LayoutConstants.ColorCyberGreen, RoundedCornerShape(18.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Default.SmartToy,
                    contentDescription = "Model icon",
                    tint = LayoutConstants.ColorCyberGreen,
                    modifier = Modifier.size(18.dp)
                )
            }
            Spacer(modifier = Modifier.width(8.dp))
        }

        Card(
            modifier = Modifier.widthIn(max = 300.dp),
            shape = RoundedCornerShape(
                topStart = 12.dp,
                topEnd = 12.dp,
                bottomStart = if (isUser) 12.dp else 2.dp,
                bottomEnd = if (isUser) 2.dp else 12.dp
            ),
            border = BorderStroke(
                width = 1.dp,
                color = if (isUser) Color(0xFF334155) else LayoutConstants.ColorCyberGreen.copy(alpha = 0.4f)
            ),
            colors = CardDefaults.cardColors(
                containerColor = if (isUser) Color(0xFF1E293B) else LayoutConstants.ColorDarkSurface
            )
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(
                    text = if (isUser) "USER PROMPT" else "NATIVE MULTI-MODAL COGNITION",
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold,
                    color = if (isUser) Color(0xFF94A3B8) else LayoutConstants.ColorCyberGreen,
                    fontFamily = FontFamily.Monospace
                )
                Spacer(modifier = Modifier.height(6.dp))
                
                // Response text content
                Text(
                    text = bubble.text,
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.White,
                    fontFamily = if (isUser) FontFamily.Default else FontFamily.Monospace,
                    fontSize = 13.sp
                )

                // Render real generated Audio Wave (if present)
                if (bubble.audioWave != null) {
                    Spacer(modifier = Modifier.height(10.dp))
                    Text(
                        text = "SYNTHESIZED WAVEFORM SAMPLES:",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = LayoutConstants.ColorCyberBlue,
                        fontFamily = FontFamily.Monospace
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    
                    // Live audio bars rendering
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(40.dp)
                            .background(Color.Black.copy(alpha = 0.4f), RoundedCornerShape(4.dp))
                            .padding(horizontal = 6.dp),
                        horizontalArrangement = Arrangement.spacedBy(2.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        // Display 20 responsive waves
                        val step = (bubble.audioWave.size / 20).coerceAtLeast(1)
                        for (i in 0 until 20) {
                            val sampleIndex = (i * step).coerceIn(0, bubble.audioWave.size - 1)
                            val amplitude = Math.abs(bubble.audioWave[sampleIndex].toInt())
                            val barHeight = (amplitude.toFloat() / 128f * 32f).coerceIn(2f, 32f).dp
                            
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .height(barHeight)
                                    .clip(RoundedCornerShape(1.dp))
                                    .background(LayoutConstants.ColorCyberBlue)
                            )
                        }
                    }
                    Text(
                        text = "Auto-regressive amplitude oscillations",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.Gray,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 9.sp
                    )
                }

                // Render real Diffusion generated 16x16 Pixel Art Grid
                if (bubble.diffuseImage != null) {
                    Spacer(modifier = Modifier.height(10.dp))
                    Text(
                        text = "LATENT RGB GRID RENDER:",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = LayoutConstants.ColorCyberPurple,
                        fontFamily = FontFamily.Monospace
                    )
                    Spacer(modifier = Modifier.height(6.dp))

                    // Draw 16x16 color block grid
                    Box(
                        modifier = Modifier
                            .size(176.dp)
                            .align(Alignment.CenterHorizontally)
                            .border(1.dp, Color(0xFF334155), RoundedCornerShape(4.dp))
                            .clip(RoundedCornerShape(4.dp))
                    ) {
                        Column(modifier = Modifier.fillMaxSize()) {
                            for (y in 0 until 16) {
                                Row(modifier = Modifier.weight(1f).fillMaxWidth()) {
                                    for (x in 0 until 16) {
                                        val pixel = bubble.diffuseImage[x][y]
                                        val color = Color(
                                            red = pixel[0],
                                            green = pixel[1],
                                            blue = pixel[2],
                                            alpha = 1f
                                        )
                                        Box(
                                            modifier = Modifier
                                                .weight(1f)
                                                .fillMaxHeight()
                                                .background(color)
                                        )
                                    }
                                }
                            }
                        }
                    }
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "Conditioned diffusion outputs via cross-attention matrix",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.Gray,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 9.sp,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.fillMaxWidth()
                    )
                }

                if (bubble.isStreaming) {
                    Spacer(modifier = Modifier.height(4.dp))
                    LinearProgressIndicator(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(2.dp),
                        color = LayoutConstants.ColorCyberGreen,
                        trackColor = Color.Transparent
                    )
                }

                // Feedback Loop/Correction section
                if (!isUser && bubble.promptSeed.isNotEmpty() && !bubble.isStreaming) {
                    Spacer(modifier = Modifier.height(12.dp))
                    HorizontalDivider(color = Color(0xFF334155))
                    Spacer(modifier = Modifier.height(6.dp))

                    if (!showCorrectionInput) {
                        TextButton(
                            onClick = { showCorrectionInput = true },
                            modifier = Modifier.align(Alignment.End)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Edit,
                                contentDescription = "Correct character prediction",
                                tint = LayoutConstants.ColorCyberGreen,
                                modifier = Modifier.size(14.dp)
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(
                                text = "CORRECT NEXT CHARACTER",
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.Bold,
                                color = LayoutConstants.ColorCyberGreen,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                    } else {
                        Column(
                            verticalArrangement = Arrangement.spacedBy(6.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = "Expected next character after seed:",
                                style = MaterialTheme.typography.labelSmall,
                                color = Color.Gray,
                                fontFamily = FontFamily.Monospace
                            )
                            
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(6.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                OutlinedTextField(
                                    value = correctiveCharInput,
                                    onValueChange = { correctiveCharInput = it.take(1) },
                                    placeholder = { Text("e.g. 'a'", color = Color.Gray) },
                                    singleLine = true,
                                    modifier = Modifier
                                        .weight(1f)
                                        .height(48.dp),
                                    shape = RoundedCornerShape(4.dp),
                                    colors = OutlinedTextFieldDefaults.colors(
                                        focusedBorderColor = LayoutConstants.ColorCyberGreen,
                                        unfocusedBorderColor = Color(0xFF334155)
                                    )
                                )

                                Button(
                                    onClick = {
                                        if (correctiveCharInput.isNotEmpty()) {
                                            onCorrectionSubmit(bubble.promptSeed, correctiveCharInput[0])
                                            correctiveCharInput = ""
                                            showCorrectionInput = false
                                        }
                                    },
                                    modifier = Modifier.height(48.dp),
                                    colors = ButtonDefaults.buttonColors(containerColor = LayoutConstants.ColorCyberGreen),
                                    shape = RoundedCornerShape(4.dp)
                                ) {
                                    Text("LEARN", style = MaterialTheme.typography.labelSmall, color = Color.Black, fontWeight = FontWeight.Bold)
                                }
                            }
                        }
                    }
                }
            }
        }

        if (isUser) {
            Spacer(modifier = Modifier.width(8.dp))
            Box(
                modifier = Modifier
                    .size(36.dp)
                    .clip(RoundedCornerShape(18.dp))
                    .background(Color(0xFF334155))
                    .border(1.dp, Color(0xFF64748B), RoundedCornerShape(18.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Default.Face,
                    contentDescription = "User icon",
                    tint = Color.White,
                    modifier = Modifier.size(18.dp)
                )
            }
        }
    }
}
