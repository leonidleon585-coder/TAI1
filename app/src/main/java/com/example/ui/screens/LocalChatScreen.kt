package com.example.ui.screens

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
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.SmartToy
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.ml.TrainerEngine
import com.example.ml.TrainingState
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

data class ChatBubble(
    val id: String,
    val sender: String, // "User" or "Offline Model"
    val text: String,
    val timestamp: Long = System.currentTimeMillis(),
    val isStreaming: Boolean = false
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LocalChatScreen(
    trainerEngine: TrainerEngine,
    state: TrainingState,
    modifier: Modifier = Modifier
) {
    val scope = rememberCoroutineScope()
    val scrollState = rememberLazyListState()

    // Conversational History State
    var chatMessages by remember {
        mutableStateOf(
            listOf(
                ChatBubble(
                    id = "welcome",
                    sender = "Offline Model",
                    text = "System cortex initialized. I am your fully autonomous Next-Character Generative Network running directly on this Snapdragon chipset. Enter a prompt to begin."
                )
            )
        )
    }

    var textInput by remember { mutableStateOf("") }
    var isThinking by remember { mutableStateOf(false) }

    // Auto-scroll to the bottom of conversation
    LaunchedEffect(chatMessages.size) {
        if (chatMessages.isNotEmpty()) {
            scrollState.animateScrollToItem(chatMessages.size - 1)
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFF030712)) // Dark cyber background
    ) {
        // Status Bar
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            shape = RoundedCornerShape(8.dp),
            border = BorderStroke(1.dp, Color(0xFF1E293B)),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF0F172A))
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Default.AutoAwesome,
                    contentDescription = "AI Status",
                    tint = Color(0xFF10B981),
                    modifier = Modifier.size(18.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Column {
                    Text(
                        text = "Active Weights Context: ${if (state.lossHistory.isEmpty()) "Untrained (Random)" else "Fine-tuned"}",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = Color.White,
                        fontFamily = FontFamily.Monospace
                    )
                    Text(
                        text = "Loss Metric: ${if (state.currentLoss > 0) String.format("%.4f", state.currentLoss) else "NaN"} • Snapdragon 8s Gen 4 Native",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.Gray,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }
        }

        // Conversational Feed
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
        ) {
            LazyColumn(
                state = scrollState,
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                items(chatMessages, key = { it.id }) { bubble ->
                    ChatBubbleRow(bubble = bubble)
                }
            }

            if (chatMessages.size == 1) {
                Text(
                    text = "Tip: Train the model in the Training Hub for 10-15 epochs to observe words and letters self-assemble from raw inputs into coherent sentences.",
                    style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center,
                    color = Color.DarkGray,
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
            color = Color(0xFF0F172A),
            tonalElevation = 4.dp,
            border = BorderStroke(1.dp, Color(0xFF1E293B))
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
                    placeholder = { Text("Feed a seed word to generating local responses...", color = Color.Gray, style = MaterialTheme.typography.bodyMedium) },
                    singleLine = true,
                    enabled = !isThinking,
                    modifier = Modifier
                        .weight(1f)
                        .height(54.dp),
                    shape = RoundedCornerShape(8.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = Color(0xFF10B981),
                        unfocusedBorderColor = Color(0xFF334155),
                        focusedLabelColor = Color(0xFF10B981)
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
                        chatMessages = chatMessages + ChatBubble(
                            id = userMessageId,
                            sender = "User",
                            text = prompt
                        )
                        textInput = ""
                        isThinking = true

                        // Stream text back character-by-character
                        scope.launch {
                            val fullResponse = trainerEngine.generateText(prompt, length = 120)
                            
                            // Initialize blank model message
                            chatMessages = chatMessages + ChatBubble(
                                id = modelMessageId,
                                sender = "Offline Model",
                                text = "",
                                isStreaming = true
                            )

                            var currentStreamedText = ""
                            for (char in fullResponse) {
                                currentStreamedText += char
                                
                                // Update message in list
                                chatMessages = chatMessages.map { bubble ->
                                    if (bubble.id == modelMessageId) {
                                        bubble.copy(text = currentStreamedText)
                                    } else {
                                        bubble
                                    }
                                }
                                delay(20) // Cadence speed for typewriter
                            }

                            // Finalize message state
                            chatMessages = chatMessages.map { bubble ->
                                if (bubble.id == modelMessageId) {
                                    bubble.copy(isStreaming = false)
                                } else {
                                    bubble
                                }
                            }
                            isThinking = false
                        }
                    },
                    enabled = textInput.trim().isNotEmpty() && !isThinking,
                    modifier = Modifier.height(54.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF10B981)),
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
fun ChatBubbleRow(bubble: ChatBubble) {
    val isUser = bubble.sender == "User"
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
                    .background(Color(0xFF10B981).copy(alpha = 0.2f))
                    .border(1.dp, Color(0xFF10B981), RoundedCornerShape(18.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Default.SmartToy,
                    contentDescription = "Model icon",
                    tint = Color(0xFF10B981),
                    modifier = Modifier.size(18.dp)
                )
            }
            Spacer(modifier = Modifier.width(8.dp))
        }

        Card(
            modifier = Modifier.widthIn(max = 280.dp),
            shape = RoundedCornerShape(
                topStart = 12.dp,
                topEnd = 12.dp,
                bottomStart = if (isUser) 12.dp else 2.dp,
                bottomEnd = if (isUser) 2.dp else 12.dp
            ),
            border = BorderStroke(
                width = 1.dp,
                color = if (isUser) Color(0xFF334155) else Color(0xFF10B981).copy(alpha = 0.4f)
            ),
            colors = CardDefaults.cardColors(
                containerColor = if (isUser) Color(0xFF1E293B) else Color(0xFF0F172A)
            )
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(
                    text = if (isUser) "USER PROMPT" else "NATIVE LLM RESPONSE",
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold,
                    color = if (isUser) Color(0xFF94A3B8) else Color(0xFF10B981),
                    fontFamily = FontFamily.Monospace
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = bubble.text,
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.White,
                    fontFamily = if (isUser) FontFamily.Default else FontFamily.Monospace
                )
                if (bubble.isStreaming) {
                    Spacer(modifier = Modifier.height(4.dp))
                    LinearProgressIndicator(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(2.dp),
                        color = Color(0xFF10B981),
                        trackColor = Color.Transparent
                    )
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
